# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    joined_str_0 = module_0.JoinedStr()
    base_import_rewrite_0 = module_1.BaseImportRewrite(joined_str_0)


def test_case_1():
    add_0 = module_0.Add()
    base_import_rewrite_0 = module_1.BaseImportRewrite(add_0)
    list_0 = [add_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    add_0 = module_0.Add()
    var_0 = module_2.dump(add_0)
    list_0 = [add_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    add_0 = module_0.Add()
    var_0 = module_2.iter_child_nodes(add_0)
    list_0 = [add_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(var_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
