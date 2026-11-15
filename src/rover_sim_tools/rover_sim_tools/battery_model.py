"""Battery model shared by ``rover_sim_tools`` and ``sandsim``.

Models a rechargeable battery pack:

    V_terminal = V_nominal - I_total * R_internal   (voltage sag)
    dE_wh/dt   = -V_terminal * I_total / 3600       (energy integration, Wh)

State of charge is derived from energy: SoC = E_wh / capacity_wh.
The pack is considered depleted when energy reaches 0 Wh or when the terminal
voltage under load drops at/below the cutoff voltage.
"""

from __future__ import annotations


class BatteryModel:
    """Simple battery model with voltage sag and energy integration."""

    def __init__(
        self,
        voltage_nominal_v: float = 12.0,
        capacity_wh: float = 360.0,
        voltage_cutoff_v: float = 10.0,
        r_internal_ohm: float = 0.05,
        initial_energy_wh: float | None = None,
    ) -> None:
        self.voltage_nominal_v = voltage_nominal_v
        self.capacity_wh = capacity_wh
        self.voltage_cutoff_v = voltage_cutoff_v
        self.r_internal_ohm = r_internal_ohm
        self.energy_wh = (
            initial_energy_wh if initial_energy_wh is not None else capacity_wh
        )
        self.total_charge_ah = capacity_wh / voltage_nominal_v

    # ------------------------------------------------------------------ #
    # electrical                                                         #
    # ------------------------------------------------------------------ #
    def terminal_voltage_v(self, total_current_a: float) -> float:
        """Terminal voltage under load (voltage sag from internal resistance).

        Raw value, not clamped.  Callers should check :meth:`is_depleted`
        before drawing current.
        """
        return self.voltage_nominal_v - total_current_a * self.r_internal_ohm

    def is_depleted(self, total_current_a: float = 0.0) -> bool:
        """True if the pack has no energy left or cannot sustain the requested
        current without dropping at/below cutoff voltage."""
        if self.energy_wh <= 0.01:  # effectively empty
            return True
        return self.terminal_voltage_v(total_current_a) <= self.voltage_cutoff_v

    def soc(self) -> float:
        """State of charge, 0.0-1.0."""
        return max(0.0, min(1.0, self.energy_wh / self.capacity_wh))

    # ------------------------------------------------------------------ #
    # energy integration                                                 #
    # ------------------------------------------------------------------ #
    def step(self, total_current_a: float, dt_s: float) -> None:
        """Advance the battery state by ``dt_s`` seconds at ``total_current_a``.

        Draws energy from the pack: dE = -V_terminal * I * dt.
        Never discharges below 0 Wh.  Stops drawing when voltage hits cutoff.
        """
        if self.energy_wh <= 0.0:
            return
        v_term = self.terminal_voltage_v(total_current_a)
        if v_term <= self.voltage_cutoff_v:
            return
        energy_joules = v_term * total_current_a * dt_s  # J = V * A * s
        energy_wh = energy_joules / 3600.0
        self.energy_wh = max(0.0, self.energy_wh - energy_wh)

    def remaining_runtime_s(self, total_current_a: float) -> float:
        """Estimated seconds of runtime at the current draw."""
        if total_current_a <= 0.0:
            return float("inf")
        v_term = self.terminal_voltage_v(total_current_a)
        if v_term <= self.voltage_cutoff_v:
            return 0.0
        power_w = v_term * total_current_a
        if power_w <= 0.0:
            return float("inf")
        return self.energy_wh * 3600.0 / power_w
