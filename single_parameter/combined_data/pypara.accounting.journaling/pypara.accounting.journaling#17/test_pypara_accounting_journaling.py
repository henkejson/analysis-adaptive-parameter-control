# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1


def test_case_0():
    pass


def test_case_1():
    bytes_0 = b"\xbb\xb1"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    bytes_0 = b"\xbb\xb1"
    none_type_0 = None
    journal_entry_0 = module_0.JournalEntry(none_type_0, none_type_0, none_type_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)


def test_case_3():
    bytes_0 = b"\xbb\xb1"
    none_type_0 = None
    journal_entry_0 = module_0.JournalEntry(none_type_0, none_type_0, none_type_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)
    journal_entry_0.validate()


def test_case_4():
    bytes_0 = b""
    none_type_0 = None
    none_type_1 = None
    journal_entry_0 = module_0.JournalEntry(none_type_0, none_type_0, none_type_1)
    var_0 = module_1.isum(bytes_0)
    none_type_2 = journal_entry_0.validate()
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)
