# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.guid as module_1
import pypara.commons.numbers as module_2


def test_case_0():
    module_0.ReadJournalEntries()


def test_case_1():
    var_0 = module_1.makeguid()
    journal_entry_0 = module_0.JournalEntry(var_0, var_0, var_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    str_0 = ""
    journal_entry_0 = module_0.JournalEntry(str_0, str_0, str_0)
    var_0 = module_2.isum(str_0)
    journal_entry_1 = journal_entry_0.post(str_0, var_0, var_0)
