"""Regression tests for the fuel-reversal / load-replan consistency scenario.

A fuel meter correction changes the load figures. The system must not patch
one ledger in isolation: either the reversal and the rebuilt load sheet are
committed together from the same data version, or the original plan is kept
for manual handling. Inventory and load records must reconcile at all times.
"""
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from rampflow.fuel_accounting import FuelTank, FuelLedger
from rampflow.load_control import (
    Compartment,
    EnvelopePoint,
    LoadControl,
    LoadItem,
)
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
D = lambda h: datetime(2026, 1, 1, h, tzinfo=UTC)


def make_ledger():
    return FuelLedger([FuelTank("T1", Decimal("10000"), Decimal("5000"), Decimal("0.8"))])


def make_load_control():
    compartments = [
        Compartment("FWD", 2.0, 5000.0, 100.0, 50.0),
        Compartment("AFT", 8.0, 5000.0, 100.0, 50.0),
    ]
    envelope = [
        EnvelopePoint(20000.0, 10.0, 40.0),
        EnvelopePoint(60000.0, 10.0, 40.0),
    ]
    return LoadControl(compartments, envelope)


class FuelReversalTests(unittest.TestCase):
    def test_reversal_cancels_temperature_corrected_amounts_exactly(self):
        ledger = make_ledger()
        delivery = ledger.record(
            "D1", "F1", "T1", Decimal("10"), Decimal("1010"),
            Decimal("25"), Decimal("0.8"), Decimal("1.2"), D(10),
        )
        self.assertNotEqual(delivery.corrected_l, Decimal("1000.000"))

        reversal = ledger.reverse("D1", "R1", D(11))

        self.assertEqual(reversal.corrected_l, -delivery.corrected_l)
        self.assertEqual(reversal.mass_kg, -delivery.mass_kg)
        self.assertEqual(reversal.amount, -delivery.amount)
        self.assertEqual(ledger.inventory_l["T1"], Decimal("5000.000"))
        summary = ledger.flight_summary("F1")
        self.assertEqual(summary["volume_l"], Decimal("0.000"))
        self.assertEqual(summary["mass_kg"], Decimal("0.000"))
        self.assertEqual(summary["amount"], Decimal("0.00"))


class LoadSheetHazmatTests(unittest.TestCase):
    def test_verify_rejects_incompatible_dangerous_goods_sharing_compartment(self):
        control = make_load_control()
        items = [
            LoadItem("H1", "cargo", 500.0, compartment_hint="FWD", dangerous_group="oxidizer"),
            LoadItem("H2", "cargo", 500.0, compartment_hint="FWD", dangerous_group="flammable"),
        ]
        # Both fit by weight, so a patched sheet could keep them together;
        # verify must catch the hazmat violation regardless.
        sheet = control.build("REG", 30000.0, 20.0, 4000.0, 2.0, 1500.0,
                              [LoadItem("B", "bags", 100.0)])
        patched = type(sheet)(
            sheet.aircraft_reg, sheet.empty_weight_kg, sheet.empty_index,
            sheet.fuel_kg, sheet.fuel_arm_m,
            {"FWD": tuple(items), "AFT": sheet.allocations["AFT"]},
            sheet.takeoff_weight_kg, sheet.takeoff_index,
            sheet.landing_weight_kg, sheet.landing_index, sheet.trim_units,
        )
        decision = control.verify(patched)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "hazmat-separation")

    def test_verify_accepts_separated_dangerous_goods(self):
        control = make_load_control()
        items = [
            LoadItem("H1", "cargo", 500.0, compartment_hint="FWD", dangerous_group="oxidizer"),
            LoadItem("H2", "cargo", 500.0, compartment_hint="AFT", dangerous_group="flammable"),
        ]
        sheet = control.build("REG", 30000.0, 20.0, 4000.0, 2.0, 1500.0, items)
        self.assertTrue(control.verify(sheet).accepted)


class FuelLoadReconciliationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ledger = make_ledger()
        self.control = make_load_control()
        self.delivery = self.ledger.record(
            "D1", "F1", "T1", Decimal("10"), Decimal("1010"),
            Decimal("25"), Decimal("0.8"), Decimal("1.2"), D(10),
        )
        self.items = [
            LoadItem("B1", "bags", 800.0),
            LoadItem("C1", "cargo", 1200.0),
        ]
        self.sheets = {}
        self.sheets["F1"] = self.control.build(
            "REG", 30000.0, 20.0, float(self.ledger.flight_totals_kg["F1"]),
            2.0, 500.0, self.items,
        )

    def _replan_after_reversal(self):
        """Build a replacement sheet from the post-reversal data version."""
        fuel_after = self.ledger.flight_totals_kg["F1"] - self.delivery.mass_kg
        try:
            sheet = self.control.build(
                "REG", 30000.0, 20.0, float(fuel_after), 4.0, 0.0, self.items,
            )
        except (RuntimeError, ValueError) as exc:
            return None, Decision(False, "replan-failed", str(exc))
        return sheet, self.control.verify(sheet)

    def test_reversal_and_replan_commit_together(self):
        workflow = RampWorkflowControl()
        new_sheet, check = self._replan_after_reversal()
        self.assertTrue(check.accepted)

        def action():
            self.ledger.reverse("D1", "R1", D(11))
            self.sheets["F1"] = new_sheet
            return new_sheet

        receipt = workflow.reconcile_fuel_load(
            "op-recon-1", "F1", D(12), action, requirements=(check,),
        )

        self.assertIs(receipt.result, new_sheet)
        self.assertEqual(self.ledger.inventory_l["T1"], Decimal("5000.000"))
        self.assertEqual(self.ledger.flight_totals_kg["F1"], Decimal("0.000"))
        self.assertIs(self.sheets["F1"], new_sheet)
        self.assertEqual(self.sheets["F1"].fuel_kg, 0.0)
        self.assertEqual(workflow.stream.head, 1)

        replay = workflow.reconcile_fuel_load(
            "op-recon-1", "F1", D(12), lambda: self.fail("re-executed"),
        )
        self.assertIs(replay, receipt)
        self.assertEqual(workflow.stream.head, 1)

    def test_failed_replan_keeps_original_plan_for_manual_handling(self):
        # Force the replan to fail: the only cargo-capable compartments cannot
        # separate these two dangerous groups once hints pin them together.
        self.items = [
            LoadItem("H1", "cargo", 500.0, compartment_hint="FWD", dangerous_group="oxidizer"),
            LoadItem("H2", "cargo", 500.0, compartment_hint="FWD", dangerous_group="flammable"),
        ]
        workflow = RampWorkflowControl()
        original_sheet = self.sheets["F1"]
        new_sheet, check = self._replan_after_reversal()
        self.assertIsNone(new_sheet)
        self.assertFalse(check.accepted)

        with self.assertRaises(WorkflowRejected):
            workflow.reconcile_fuel_load(
                "op-recon-2", "F1", D(12),
                lambda: self.fail("action must not run"),
                requirements=(check,),
            )

        # Neither ledger moved: original plan retained pending manual handling.
        self.assertIs(self.sheets["F1"], original_sheet)
        self.assertEqual(self.ledger.inventory_l["T1"], Decimal("4009.411"))
        self.assertEqual(self.ledger.flight_totals_kg["F1"], self.delivery.mass_kg)
        self.assertEqual(workflow.stream.head, 0)
        self.assertEqual(workflow.receipts, {})

    def test_domain_failure_publishes_no_event(self):
        workflow = RampWorkflowControl()

        def action():
            raise RuntimeError("load sheet rebuild failed")

        with self.assertRaises(RuntimeError):
            workflow.reconcile_fuel_load("op-recon-3", "F1", D(12), action)
        self.assertEqual(workflow.stream.head, 0)
        self.assertEqual(workflow.receipts, {})


if __name__ == "__main__":
    unittest.main()
