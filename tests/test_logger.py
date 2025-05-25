import unittest
import csv
import os
import sys
from datetime import datetime

class TableTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super(TableTestResult, self).__init__(*args, **kwargs)
        self.test_results = []  # Container for table rows

    def addSuccess(self, test):
        super().addSuccess(test)
        # Try to get expected and actual outputs from test attributes.
        expected = getattr(test, "expected_output", "N/A")
        self.test_results.append({
            "Nome Test": str(test),
            "Data": datetime.now().isoformat(),
            "Output Atteso": expected,
            "Output Effettivo": expected,
            "Risultato": "Successo"
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        # For failures, you can capture expected/actual from attributes if set, or leave as "N/A".
        expected = getattr(test, "expected_output", "N/A")
        # Use the formatted exception information as the actual output.
        actual = self._exc_info_to_string(err, test)
        self.test_results.append({
            "Nome Test": str(test),
            "Data": datetime.now().isoformat(),
            "Output Atteso": expected,
            "Output Effettivo": actual,
            "Risultato": "Fallimento"
        })

    def addError(self, test, err):
        super().addError(test, err)
        expected = getattr(test, "expected_output", "N/A")
        actual = self._exc_info_to_string(err, test)
        self.test_results.append({
            "Nome Test": str(test),
            "Data": datetime.now().isoformat(),
            "Output Atteso": expected,
            "Output Effettivo": actual,
            "Risultato": "Errore"
        })

# Custom TestRunner that uses TableTestResult and writes out a CSV file.
class TableTestRunner(unittest.TextTestRunner):
    def __init__(self, filename, stream=None, descriptions=True, verbosity=1):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []

        folder = "TestResults"
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.filename = os.path.join(folder, filename)

    def _makeResult(self):
        return TableTestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        result = super().run(test)
        self._write_results(result.test_results)
        return result

    def _write_results(self, test_results):

        with open(self.filename, "w", newline='', encoding="utf-8") as csvfile:
            fieldnames = ["Nome Test", "Data", "Output Atteso", "Output Effettivo", "Risultato"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in test_results:
                writer.writerow(row)
        print(f"Test results have been written to: {self.filename}")