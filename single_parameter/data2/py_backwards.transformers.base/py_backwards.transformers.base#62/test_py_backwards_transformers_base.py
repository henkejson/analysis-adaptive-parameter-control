# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    store_0 = module_0.Store()
    base_import_rewrite_0 = module_1.BaseImportRewrite(store_0)


def test_case_1():
    function_type_0 = module_0.FunctionType()
    base_import_rewrite_0 = module_1.BaseImportRewrite(function_type_0)
    list_0 = [function_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    str_0 = "s;Xwh'8F-N4$\x0bt\x0c_H{.s"
    base_import_rewrite_0 = module_1.BaseImportRewrite(str_0)
    list_0 = [str_0, str_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit(import_from_0)


def test_case_4():
    str_0 = "skwh8F-N\x0bt\x0c_H{5s"
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    attribute_0 = module_0.Attribute(**dict_0)
    var_0 = module_2.iter_child_nodes(attribute_0)
    list_0 = [var_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(attribute_0)
    var_1 = base_import_rewrite_0.visit(import_from_0)
