import os
import sys
import unittest
from unittest.mock import patch
from test_logger import TableTestRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utility.cmdlineManagement.PlaceholderUserManager import PlaceholderUserManager  

class TestPlaceholderUserManager(unittest.TestCase):
    # SUP1: AF1 - SU1: Valid alias_file and user selects "Eliminare Utente generico"
    @patch('inquirer.prompt', return_value={'operation': 'Eliminare Utente generico'})
    def test_selection_eliminare(self, mock_prompt):
        manager = PlaceholderUserManager(alias_file='dummy_path')
        result = manager.selection()
        self.assertEqual(result, ("other", True))

    # SUP2: AF1 - SU2: Valid alias_file and user selects "Soprannominare Utente generico"
    @patch('builtins.input', return_value='NewUserName')
    @patch('inquirer.prompt', return_value={'operation': 'Soprannominare Utente generico'})
    def test_selection_soprannominare(self, mock_prompt, mock_input):
        manager = PlaceholderUserManager(alias_file='dummy_path')
        result = manager.selection()
        self.assertEqual(result, ("NewUserName", True))

    # SUP3: AF1 - SU3: Valid alias_file and user selects "Procedere senza modifiche"
    @patch('inquirer.prompt', return_value={'operation': 'Procedere senza modifiche'})
    def test_selection_procedere(self, mock_prompt):
        manager = PlaceholderUserManager(alias_file='dummy_path')
        result = manager.selection()
        self.assertEqual(result, ("other", False))

    # SUP4: AF2: alias_file is None, so selection returns ("other", False) immediately
    def test_selection_no_alias_file(self):
        manager = PlaceholderUserManager(alias_file=None)
        result = manager.selection()
        self.assertEqual(result, ("other", False))

if __name__ == '__main__':
    unittest.main(testRunner=TableTestRunner("PlaceholderUserManager.csv"))