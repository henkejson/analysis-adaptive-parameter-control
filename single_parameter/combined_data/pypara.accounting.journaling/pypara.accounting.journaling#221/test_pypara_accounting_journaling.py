# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import datetime as module_0
import pypara.accounting.journaling as module_1


def test_case_0():
    module_0.datetime()


def test_case_1():
    str_0 = ">,W1@_rFm M"
    journal_entry_0 = module_1.JournalEntry(str_0, str_0, str_0)
    none_type_0 = journal_entry_0.validate()
