# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1


def test_case_0():
    pass


def test_case_1():
    direction_0 = module_0.Direction.DEC
    journal_entry_0 = module_0.JournalEntry(direction_0, direction_0, direction_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    none_type_0 = None
    str_0 = ""
    journal_entry_0 = module_0.JournalEntry(str_0, str_0, str_0)
    var_0 = module_1.isum(str_0, none_type_0)
    journal_entry_1 = journal_entry_0.post(var_0, str_0, var_0)


def test_case_3():
    str_0 = ""
    journal_entry_0 = module_0.JournalEntry(str_0, str_0, str_0)
    bytes_0 = b"o|\x9f\x99"
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(journal_entry_0, var_0, var_0)


def test_case_4():
    str_0 = "P"
    journal_entry_0 = module_0.JournalEntry(str_0, str_0, str_0)
    bytes_0 = b"o|\xc6"
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(journal_entry_0, var_0, var_0)
    journal_entry_0.validate()
