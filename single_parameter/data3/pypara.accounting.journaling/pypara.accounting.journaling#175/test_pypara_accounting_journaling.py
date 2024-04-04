# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.accounting.accounts as module_1
import pypara.commons.numbers as module_2


def test_case_0():
    pass


def test_case_1():
    bool_0 = False
    journal_entry_0 = module_0.JournalEntry(bool_0, bool_0, bool_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    account_type_0 = module_1.AccountType.LIABILITIES
    journal_entry_0 = module_0.JournalEntry(
        account_type_0, account_type_0, account_type_0
    )
    bytes_0 = b"\xb4\xce\x8bv\x1f\xb2&C\xbcS\xea\xc3\xe1\xe0\xa6\x86{\xa2"
    var_0 = module_2.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)


def test_case_3():
    account_type_0 = module_1.AccountType.LIABILITIES
    journal_entry_0 = module_0.JournalEntry(
        account_type_0, account_type_0, account_type_0
    )
    bytes_0 = b"\xb4\xce\x8bv\x1f\xb2&C\xbcS\xea\xc3\xe1\xe0\xa6\x86{\xa2"
    var_0 = module_2.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)
    journal_entry_0.validate()


def test_case_4():
    account_type_0 = module_1.AccountType.LIABILITIES
    journal_entry_0 = module_0.JournalEntry(
        account_type_0, account_type_0, account_type_0
    )
    bytes_0 = b""
    none_type_0 = journal_entry_0.validate()
    var_0 = module_2.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, var_0, var_0)
    none_type_1 = journal_entry_1.validate()
