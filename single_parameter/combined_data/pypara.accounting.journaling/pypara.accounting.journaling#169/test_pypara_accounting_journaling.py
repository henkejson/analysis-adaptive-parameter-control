# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1
import pypara.accounting.accounts as module_2


def test_case_0():
    module_0.ReadJournalEntries()


def test_case_1():
    bytes_0 = b"uO\xa7"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    bytes_0 = b"\x08#\x99\xb9\xc4~i\xaeW\x95"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)


def test_case_3():
    bytes_0 = b"\x94,\xa1"
    none_type_0 = None
    journal_entry_0 = module_0.JournalEntry(none_type_0, bytes_0, bytes_0)
    none_type_1 = journal_entry_0.validate()
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(none_type_1, var_0, var_0)
    journal_entry_1.validate()


def test_case_4():
    bytes_0 = b""
    account_type_0 = module_2.AccountType.LIABILITIES
    var_0 = account_type_0.__hash__()
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = None
    journal_entry_1 = module_0.JournalEntry(var_0, var_0, none_type_0)
    none_type_1 = journal_entry_0.validate()
    var_1 = var_0.__hash__()
    var_2 = module_1.isum(bytes_0)
    journal_entry_2 = journal_entry_0.post(var_0, account_type_0, var_2)
    none_type_2 = journal_entry_0.validate()
    module_1.isum(var_0, account_type_0)
