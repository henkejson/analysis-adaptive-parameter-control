# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(none_type_0)


def test_case_1():
    l_shift_0 = module_1.LShift()
    base_import_rewrite_0 = module_0.BaseImportRewrite(l_shift_0)
    list_0 = [l_shift_0, l_shift_0, l_shift_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit(import_from_0)


def test_case_2():
    l_shift_0 = module_1.LShift()
    base_import_rewrite_0 = module_0.BaseImportRewrite(l_shift_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
