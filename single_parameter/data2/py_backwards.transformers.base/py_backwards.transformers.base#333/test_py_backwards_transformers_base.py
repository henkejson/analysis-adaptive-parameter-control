# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    name_constant_0 = module_0.NameConstant()
    base_import_rewrite_0 = module_1.BaseImportRewrite(name_constant_0)


def test_case_1():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [base_import_rewrite_0]
    list_1 = [list_0, list_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = []
    list_1 = [list_0, list_0]
    import_from_0 = module_0.ImportFrom(*list_1)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
