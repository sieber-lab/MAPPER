"""Regression checks for ambiguous names and shared measured protein groups."""

import unittest
from collections import defaultdict

from mapper.analysis.identifiers import Reference


class IdentifierIdentityTests(unittest.TestCase):
    def setUp(self):
        self.reference = Reference.__new__(Reference)
        self.reference.targets = {"ribC", "ribE", "tufA", "tufB"}
        self.reference.names = defaultdict(
            set,
            {
                "ribC": {"ribC"},
                "ribE": {"ribE"},
                "ribH": {"ribC", "ribE"},
                "tufA": {"tufA"},
                "tufB": {"tufB"},
            },
        )
        self.reference.accessions = defaultdict(set, {"RISB": {"ribE"}, "TUF": {"tufA", "tufB"}})
        self.reference.entries = defaultdict(set)

    def test_accession_disambiguates_historical_name(self):
        result = self.reference.resolve("ribE;ribH", "RISB", allow_multiple=True)
        self.assertEqual(result["targets"], ["ribE"])

    def test_unsupported_ambiguous_alias_stays_unresolved(self):
        self.assertEqual(self.reference.resolve("ribH")["targets"], [])

    def test_shared_group_not_arbitrarily_assigned(self):
        self.assertEqual(self.reference.resolve("tufA;tufB", "TUF")["targets"], [])
        self.assertEqual(
            self.reference.resolve("tufA;tufB", "TUF", allow_multiple=True)["targets"],
            ["tufA", "tufB"],
        )

    def test_conflicting_gene_and_protein_are_not_accepted(self):
        result = self.reference.resolve("ribC", "RISB")
        self.assertEqual(result["rule"], "gene_accession_conflict")
        self.assertEqual(result["targets"], [])


if __name__ == "__main__":
    unittest.main()
