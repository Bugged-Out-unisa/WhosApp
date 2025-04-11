import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'utility', 'dataset')))

from rawDataReader import rawDataReader

class TestRawDataReader(unittest.TestCase):

    # LF1: Single valid file – reading is successful
    @patch("os.path.isfile")
    @patch("os.listdir")
    def test_LF1_single_file(self, mock_listdir, mock_isfile):
        # Simulate one file in the directory.
        mock_listdir.return_value = ["file1.txt"]
        # Assume every file reported is valid.
        mock_isfile.return_value = True

        # Patch the built-in open to simulate file content.
        m = mock_open(read_data="Content of file1")
        with patch("builtins.open", m):
            rdr = rawDataReader("dummy/path")
            result = rdr.read_all_files()
            self.assertEqual(result, ["Content of file1"])

    # LF2: Multiple valid files – reading is successful
    @patch("os.path.isfile")
    @patch("os.listdir")
    def test_LF2_multiple_files(self, mock_listdir, mock_isfile):
        # Simulate two files in the directory.
        mock_listdir.return_value = ["file1.txt", "file2.txt"]
        mock_isfile.return_value = True

        # To handle multiple files, we define a side effect function
        # that returns a different mock for each file opened.
        def open_side_effect(file, mode='r', encoding=None):
            filename = os.path.basename(file)
            file_content = f"Content of {filename}"
            file_mock = mock_open(read_data=file_content).return_value
            file_mock.__iter__.return_value = file_content.splitlines()
            return file_mock

        with patch("builtins.open", side_effect=open_side_effect):
            rdr = rawDataReader("dummy/path")
            result = rdr.read_all_files()
            expected = ["Content of file1.txt", "Content of file2.txt"]
            self.assertEqual(result, expected)

    # LF3: Invalid file path – should raise an error (e.g., FileNotFoundError)
    @patch("os.listdir")
    def test_LF3_invalid_path(self, mock_listdir):
        # Simulate that os.listdir fails because the path is invalid.
        mock_listdir.side_effect = FileNotFoundError("Invalid path")
        rdr = rawDataReader("invalid/path")
        with self.assertRaises(FileNotFoundError):
            rdr.read_all_files()

    # LF4: Empty folder – no files found, should raise an error
    @patch("os.path.isfile")
    @patch("os.listdir")
    def test_LF4_empty_folder(self, mock_listdir, mock_isfile):
        # Simulate an empty directory.
        mock_listdir.return_value = []
        rdr = rawDataReader("dummy/path")
        with self.assertRaises(Exception) as context:
            rdr.read_all_files()
        self.assertIn("Nessun file di testo trovato", str(context.exception))

if __name__ == '__main__':
    unittest.main()
