# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1


def test_case_0():
    pass


def test_case_1():
    bool_0 = True
    journal_entry_0 = module_0.JournalEntry(bool_0, bool_0, bool_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    bytes_0 = b"g\x15\xa7\xef\xc7\xd0C\x99\x8d\xc0\x84\xf9\xee\xbd\r\xef\x9bww\xb7"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, bytes_0, var_0)


def test_case_3():
    bytes_0 = b"g\x15\xa7\xef\xc7\xd0C\x99\x8d\xc0\x84\xf9\xee\xbd\r\xef\x9bww\xb7"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, bytes_0, var_0)
    journal_entry_0.validate()
