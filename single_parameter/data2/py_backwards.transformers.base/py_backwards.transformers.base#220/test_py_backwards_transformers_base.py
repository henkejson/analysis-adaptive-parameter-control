# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1


def test_case_0():
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(none_type_0)


def test_case_1():
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0, none_type_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    str_0 = "F="
    import_from_0 = module_1.ImportFrom(*str_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(import_from_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    str_0 = ""
    list_0 = [str_0, str_0, str_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(import_from_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
