# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(none_type_0)


def test_case_1():
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    str_0 = "}1Upw\r^"
    dict_0 = {str_0: list_0, str_0: list_0, str_0: none_type_0}
    import_from_0 = module_1.ImportFrom(*list_0, **dict_0)
    async_function_def_0 = module_1.AsyncFunctionDef(**dict_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(async_function_def_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    str_0 = "urllib.robotparser"
    dict_0 = {str_0: str_0}
    import_from_0 = module_1.ImportFrom(*dict_0, **dict_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(dict_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    ann_assign_0 = module_1.AnnAssign()
    var_0 = module_2.walk(ann_assign_0)
    list_0 = [var_0, var_0, var_0]
    str_0 = "}1Upw\r^"
    dict_0 = {str_0: list_0, str_0: list_0, str_0: var_0}
    import_from_0 = module_1.ImportFrom(*list_0, **dict_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(var_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    ann_assign_0 = module_1.AnnAssign()
    var_0 = module_2.iter_child_nodes(ann_assign_0)
    list_0 = [var_0, var_0, var_0]
    dict_0 = {}
    import_from_0 = module_1.ImportFrom(*list_0, **dict_0)
    base_import_rewrite_0 = module_0.BaseImportRewrite(ann_assign_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
