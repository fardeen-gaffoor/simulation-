/*
 * Copyright (C) 2026 Caterpillar Autonomy Challenge team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * sand_traction_plugin.cpp
 *
 * Gazebo Harmonic (gz-sim 8) system plugin that applies sand traction forces
 * to rover wheels using a seedable lookup table exported by the sandsim
 * terramechanics tool (Bekker-Wong model).
 *
 * The plugin reads a load x slip lookup table (drawbar pull and torque vs slip)
 * and applies the corresponding longitudinal force and resisting torque to
 * each wheel joint each simulation step.
 *
 * Usage (in arena.sdf):
 *   <plugin filename="gz-sim-sand-traction-system"
 *            name="gz::sim::systems::SandTractionPlugin">
 *     <lookup_table_csv>/path/to/sand_traction_lookup.csv</lookup_table_csv>
 *     <wheel_joint_names>
 *       <joint>wheel_front_left_joint</joint>
 *       ...
 *     </wheel_joint_names>
 *     <mass_kg>50.0</mass_kg>
 *     <radius_m>0.10</radius_m>
 *     <width_m>0.06</width_m>
 *     <longitudinal_friction_mu>0.9</longitudinal_friction_mu>
 *     <lateral_friction_mu>0.3</lateral_friction_mu>
 *   </plugin>
 */

#include <gz/sim/System.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/plugin/Register.hh>
#include <gz/common/Console.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>

#include <sdf/Element.hh>

using namespace gz::common;

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace gz
{
namespace sim
{
namespace systems
{

// Forward declare registration macro
// GZ_ADD_PLUGIN(SandTractionPlugin, ISystemPreUpdate, ISystemConfigure)

class SandTractionPlugin : public System,
                         public ISystemConfigure,
                         public ISystemPreUpdate
{
public:
  SandTractionPlugin() = default;
  ~SandTractionPlugin() override = default;

  // Configuration
  void Configure(const Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 EntityComponentManager &_ecm,
                 EventManager &_eventMgr) override;

  // Pre-update: apply forces before physics step
  void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override;

private:
  // Lookup table entry
  struct LookupEntry
  {
    double load_N;
    double slip;
    double drawbar_pull_N;
    double torque_Nm;
    double sinkage_m;
  };

  // Bilinear interpolation over (load, slip) grid
  double interpPull(double load, double slip) const;
  double interpTorque(double load, double slip) const;

  // Parse lookup table CSV
  bool loadTable(const std::string &_path);

  // Compute per-wheel vertical load (static + longitudinal transfer)
  double computeWheelLoad(double _mass, double _cg_height, double _wheelbase,
                          double _cg_offset, double _long_accel,
                          bool _is_front) const;

  // Parameters
  std::string lookupTablePath;
  std::vector<std::string> wheelJointNames;
  double massKg = 50.0;
  double radiusM = 0.10;
  double widthM = 0.06;
  double muLong = 0.9;
  double muLat = 0.3;

  // State
  bool configured = false;
  std::vector<LookupEntry> table;
  std::vector<double> uniqueLoads;
  std::vector<double> uniqueSlips;
  std::vector<std::vector<double>> pullGrid;  // [slipIdx][loadIdx]
  std::vector<std::vector<double>> torqueGrid; // [slipIdx][loadIdx]
  Entity modelEntity{kNullEntity};
};

// Register the plugin
GZ_ADD_PLUGIN(SandTractionPlugin,
              ::gz::sim::v8::ISystemPreUpdate,
              ::gz::sim::v8::ISystemConfigure);

///////////////////////////////////////////////////////////////
void SandTractionPlugin::Configure(const Entity &_entity,
                                   const std::shared_ptr<const sdf::Element> &_sdf,
                                   EntityComponentManager &_ecm,
                                   EventManager & /*_eventMgr*/)
{
  modelEntity = _entity;

  if (_sdf->HasElement("lookup_table_csv"))
  {
    lookupTablePath = _sdf->Get<std::string>("lookup_table_csv");
  }
  else
  {
    ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Missing <lookup_table_csv> element\n";
    return;
  }

  if (_sdf->HasElement("wheel_joint_names"))
  {
    auto nonConst = const_cast<sdf::Element*>(_sdf.get());
    auto wheelElem = nonConst->GetElement("wheel_joint_names");
    while (wheelElem)
    {
      auto val = wheelElem->GetValue();
      std::string jname;
      if (val && val->Get(jname))
      {
        wheelJointNames.push_back(jname);
      }
      wheelElem = wheelElem->GetNextElement("joint");
    }
  }
  else
  {
    ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Missing <wheel_joint_names> element\n";
    return;
  }

  if (_sdf->HasElement("mass_kg"))
    massKg = _sdf->Get<double>("mass_kg");
  if (_sdf->HasElement("radius_m"))
    radiusM = _sdf->Get<double>("radius_m");
  if (_sdf->HasElement("width_m"))
    widthM = _sdf->Get<double>("width_m");
  if (_sdf->HasElement("longitudinal_friction_mu"))
    muLong = _sdf->Get<double>("longitudinal_friction_mu");
  if (_sdf->HasElement("lateral_friction_mu"))
    muLat = _sdf->Get<double>("lateral_friction_mu");

  // Load lookup table
  if (!loadTable(lookupTablePath))
  {
    ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Failed to load lookup table: "
          << lookupTablePath << "\n";
    return;
  }

  configured = true;
  ::gz::common::Console::msg() << "[SandTractionPlugin] Configured with " << wheelJointNames.size()
        << " wheels, lookup table: " << lookupTablePath << "\n";
}

/////////////////////////////////////////////////////////////////
bool SandTractionPlugin::loadTable(const std::string &_path)
{
  std::ifstream file(_path);
  if (!file.is_open())
  {
    ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Cannot open: " << _path << "\n";
    return false;
  }

  std::string line;
  std::getline(file, line); // skip header

  table.clear();
  while (std::getline(file, line))
  {
    if (line.empty())
      continue;
    LookupEntry e;
    char comma;
    double dummy;
    // Parse: load_N,slip,drawbar_pull_N,torque_Nm,sinkage_m
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf",
                    &e.load_N, &e.slip, &e.drawbar_pull_N,
                    &e.torque_Nm, &e.sinkage_m) == 5)
    {
      table.push_back(e);
    }
  }

  if (table.empty())
  {
    ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Lookup table is empty: " << _path << "\n";
    return false;
  }

  // Build unique sorted loads and slips
  std::set<double> loadSet, slipSet;
  for (const auto &e : table)
  {
    loadSet.insert(e.load_N);
    slipSet.insert(e.slip);
  }
  uniqueLoads.assign(loadSet.begin(), loadSet.end());
  uniqueSlips.assign(slipSet.begin(), slipSet.end());

  size_t nSlip = uniqueSlips.size();
  size_t nLoad = uniqueLoads.size();

  pullGrid.assign(nSlip, std::vector<double>(nLoad, 0.0));
  torqueGrid.assign(nSlip, std::vector<double>(nLoad, 0.0));

  // Build maps from value to index
  std::map<double, size_t> loadIdx, slipIdx;
  for (size_t i = 0; i < nLoad; ++i)
    loadIdx[uniqueLoads[i]] = i;
  for (size_t i = 0; i < nSlip; ++i)
    slipIdx[uniqueSlips[i]] = i;

  for (const auto &e : table)
  {
    size_t si = slipIdx[e.slip];
    size_t li = loadIdx[e.load_N];
    pullGrid[si][li] = e.drawbar_pull_N;
    torqueGrid[si][li] = e.torque_Nm;
  }

  ::gz::common::Console::msg() << "[SandTractionPlugin] Loaded " << table.size()
        << " table entries (" << nLoad << " loads x " << nSlip << " slips)\n";
  return true;
}

/////////////////////////////////////////////////////////////////
double SandTractionPlugin::interpPull(double _load, double _slip) const
{
  // Clamp
  double load = std::clamp(_load, uniqueLoads.front(), uniqueLoads.back());
  double slip = std::clamp(_slip, uniqueSlips.front(), uniqueSlips.back());

  // Find bracketing indices
  size_t li = 0;
  while (li + 1 < uniqueLoads.size() && uniqueLoads[li + 1] <= load)
    ++li;
  size_t si = 0;
  while (si + 1 < uniqueSlips.size() && uniqueSlips[si + 1] <= slip)
    ++si;

  double u = (load - uniqueLoads[li]) /
             (uniqueLoads[li + 1] - uniqueLoads[li] + 1e-12);
  double v = (slip - uniqueSlips[si]) /
             (uniqueSlips[si + 1] - uniqueSlips[si] + 1e-12);

  double v00 = pullGrid[si][li];
  double v01 = pullGrid[si][li + 1];
  double v10 = pullGrid[si + 1][li];
  double v11 = pullGrid[si + 1][li + 1];

  return v00 * (1 - u) * (1 - v) + v01 * u * (1 - v) +
         v10 * (1 - u) * v + v11 * u * v;
}

/////////////////////////////////////////////////////////////////
double SandTractionPlugin::interpTorque(double _load, double _slip) const
{
  double load = std::clamp(_load, uniqueLoads.front(), uniqueLoads.back());
  double slip = std::clamp(_slip, uniqueSlips.front(), uniqueSlips.back());

  size_t li = 0;
  while (li + 1 < uniqueLoads.size() && uniqueLoads[li + 1] <= load)
    ++li;
  size_t si = 0;
  while (si + 1 < uniqueSlips.size() && uniqueSlips[si + 1] <= slip)
    ++si;

  double u = (load - uniqueLoads[li]) /
             (uniqueLoads[li + 1] - uniqueLoads[li] + 1e-12);
  double v = (slip - uniqueSlips[si]) /
             (uniqueSlips[si + 1] - uniqueSlips[si] + 1e-12);

  double v00 = torqueGrid[si][li];
  double v01 = torqueGrid[si][li + 1];
  double v10 = torqueGrid[si + 1][li];
  double v11 = torqueGrid[si + 1][li + 1];

  return v00 * (1 - u) * (1 - v) + v01 * u * (1 - v) +
         v10 * (1 - u) * v + v11 * u * v;
}

/////////////////////////////////////////////////////////////////
double SandTractionPlugin::computeWheelLoad(double _mass, double _cgHeight,
                                            double _wheelbase,
                                            double _cgOffset,
                                            double _longAccel,
                                            bool _isFront) const
{
  double wTotal = _mass * 9.81;
  double halfWb = _wheelbase / 2.0;

  // Static load distribution (front/rear based on CG offset)
  double frontAxleLoad =
      (_wheelbase / 2.0 - _cgOffset) / _wheelbase * wTotal / 2.0;
  double rearAxleLoad = wTotal / 2.0 - frontAxleLoad;

  double axleLoad = _isFront ? frontAxleLoad : rearAxleLoad;
  double wheelLoad = axleLoad / 2.0; // per wheel (left/right split)

  // Longitudinal load transfer during acceleration
  double transfer = _mass * _longAccel * _cgHeight / 2.0;
  if (_isFront)
    wheelLoad += transfer; // front gets more load during acceleration
  else
    wheelLoad -= transfer;

  return std::max(wheelLoad, 0.0);
}

/////////////////////////////////////////////////////////////////
void SandTractionPlugin::PreUpdate(const UpdateInfo &_info,
                                   EntityComponentManager &_ecm)
{
  if (!configured)
    return;

  // Only apply when not paused
  if (_info.paused)
    return;

  double dt = _info.dt.count();

  // Estimate longitudinal acceleration from wheel velocities change
  // (simple estimate: use current wheel speeds)
  double longAccel = 0.0;

  // Find model links to get mass/inertial if needed
  // For now, use configured mass

  for (const auto &jn : wheelJointNames)
  {
    // Find joint entity by name
    Entity jointEntity{kNullEntity};
    for (auto entity : _ecm.Descendants(kNullEntity))
    {
      auto *jointComp = _ecm.Component<components::Joint>(entity);
      auto *nameComp = _ecm.Component<components::Name>(entity);
      if (jointComp && nameComp && nameComp->Data() == jn)
      {
        jointEntity = entity;
        break;
      }
    }

    if (jointEntity == kNullEntity)
    {
      ::gz::common::Console::err(__FILE__, __LINE__) << "[SandTractionPlugin] Joint not found: " << jn << "\n";
      continue;
    }

    // Get current joint velocity
    double jointVel = 0.0;
    auto *jointVelComp = _ecm.Component<components::JointVelocity>(jointEntity);
    if (jointVelComp)
    {
      jointVel = jointVelComp->Data()[0];
    }

    // Get parent link (wheel link) to compute load
    // For load estimation, use configured mass evenly distributed
    // In a more refined version, we'd read link inertial mass
    double wheelLoad = computeWheelLoad(massKg, 0.25 /* cgHeight */,
                                        0.65 /* wheelbase */,
                                        0.0 /* cgOffset */,
                                        longAccel,
                                        /* front */ jn.find("front") != std::string::npos);

    // Compute slip: for a driven wheel, slip = (omega*r - vx) / max(vx, omega*r)
    // We approximate vehicle speed from wheel average or use a simple model
    // Here: use wheel angular velocity * radius as the "free rolling" speed
    // slip is defined against the rod-tip radius
    double wheelSpeed = jointVel * radiusM; // m/s at rim

    // Estimate vehicle forward speed (use average of all wheels or a state var)
    // For simplicity, use the current wheel speed as reference (slip ~ 0 when
    // rolling freely). A real implementation would track vehicle velocity.
    double vehicleSpeed = wheelSpeed; // approximation

    double slip = 0.0;
    double wheelRimSpeed = std::abs(jointVel) * radiusM;
    if (wheelRimSpeed > 0.01)
    {
      slip = (wheelRimSpeed - vehicleSpeed) / wheelRimSpeed;
      slip = std::clamp(slip, -0.9, 0.9);
    }

    // Look up traction
    double pull = interpPull(wheelLoad, slip);
    double torque = interpTorque(wheelLoad, slip);

    // Apply longitudinal force as joint force (torque about wheel axis)
    // The drawbar pull is a force at the contact patch; torque at the axle
    // = pull * radius. We command joint torque.
    double cmdTorque = torque; // N.m about wheel axle

    // Clamp by friction cone (optional safety)
    double maxTorque = muLong * wheelLoad * radiusM;
    cmdTorque = std::clamp(cmdTorque, -maxTorque, maxTorque);

    // Set joint force command
    auto *forceCmd = _ecm.Component<components::JointForceCmd>(jointEntity);
    if (!forceCmd)
    {
      _ecm.CreateComponent(jointEntity, components::JointForceCmd{std::vector<double>{cmdTorque}});
    }
    else
    {
      forceCmd->Data()[0] = cmdTorque;
    }
  }
}

} // namespace systems
} // namespace sim
} // namespace gz
