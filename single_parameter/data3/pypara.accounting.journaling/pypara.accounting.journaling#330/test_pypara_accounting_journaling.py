# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0


def test_case_0():
    module_0.ReadJournalEntries()


def test_case_1():
    var_0 = module_0.Direction.INC
    journal_entry_0 = module_0.JournalEntry(var_0, var_0, var_0)
    none_type_0 = journal_entry_0.validate()
