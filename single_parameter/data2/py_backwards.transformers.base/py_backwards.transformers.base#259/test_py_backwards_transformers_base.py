# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    not_eq_0 = module_0.NotEq()
    base_node_transformer_0 = module_1.BaseNodeTransformer(not_eq_0)


def test_case_1():
    list_comp_0 = module_0.ListComp()
    list_0 = [list_comp_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_comp_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    l_shift_0 = module_0.LShift()
    list_comp_0 = module_0.ListComp()
    base_import_rewrite_0 = module_1.BaseImportRewrite(list_comp_0)
    none_type_0 = None
    base_import_rewrite_1 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    l_shift_0 = module_0.LShift()
    yield_from_0 = module_2.iter_fields(l_shift_0)
    list_0 = [l_shift_0, yield_from_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    l_shift_0 = module_0.LShift()
    var_0 = module_2.walk(l_shift_0)
    list_0 = [l_shift_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(l_shift_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
