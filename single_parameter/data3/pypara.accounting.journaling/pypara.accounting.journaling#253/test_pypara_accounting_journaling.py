# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1


def test_case_0():
    pass


def test_case_1():
    bytes_0 = b"\xf5"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    bytes_0 = b"\xf6\xdb\xea^I\x0cQ\xc2-\xd5/\xe0\xad"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.__eq__(bytes_0)
    set_0 = set()
    var_0 = module_1.isum(set_0, set_0)
    journal_entry_1 = journal_entry_0.post(bytes_0, set_0, var_0)


def test_case_3():
    bytes_0 = b"\x98o^u\x91*\xd7g\xbcQj\x90"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(journal_entry_0, journal_entry_0, var_0)


def test_case_4():
    bytes_0 = b"\x98o^u\x91*\xd7g\xbcQj\x90"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(journal_entry_0, journal_entry_0, var_0)
    journal_entry_0.validate()
