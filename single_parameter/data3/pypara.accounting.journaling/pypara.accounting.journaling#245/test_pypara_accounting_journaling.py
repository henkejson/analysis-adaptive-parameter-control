# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pypara.accounting.journaling as module_0
import pypara.commons.numbers as module_1
import dataclasses as module_2


def test_case_0():
    module_0.ReadJournalEntries()


def test_case_1():
    bytes_0 = b"Ua\xf7\x8d\x84H^Ob,\x8d\xd1j\xbefP\x1c"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.validate()


def test_case_2():
    bytes_0 = b"d(\x91Nw\xfa;\xe0\xb9d\xfa{d\x07<G"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, bytes_0, var_0)


def test_case_3():
    bytes_0 = b"d(\x91Nw\xfa;\xe0\xb9d\xfa{d\x07<G"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, bytes_0, var_0)
    journal_entry_0.validate()


def test_case_4():
    bytes_0 = b"dU\xfb|(\xdcMImC\xfb\x15"
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = module_2.field(default=journal_entry_0)
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, none_type_0, var_0)
    journal_entry_0.validate()


def test_case_5():
    bytes_0 = b""
    journal_entry_0 = module_0.JournalEntry(bytes_0, bytes_0, bytes_0)
    none_type_0 = journal_entry_0.validate()
    var_0 = module_1.isum(bytes_0)
    journal_entry_1 = journal_entry_0.post(var_0, bytes_0, var_0)
    none_type_1 = journal_entry_1.validate()
