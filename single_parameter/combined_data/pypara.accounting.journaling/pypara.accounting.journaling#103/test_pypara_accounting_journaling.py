# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.accounts as module_0
import pypara.accounting.journaling as module_1


def test_case_0():
    pass


def test_case_1():
    account_type_0 = module_0.AccountType.LIABILITIES
    journal_entry_0 = module_1.JournalEntry(
        account_type_0, account_type_0, account_type_0
    )
    none_type_0 = journal_entry_0.validate()
