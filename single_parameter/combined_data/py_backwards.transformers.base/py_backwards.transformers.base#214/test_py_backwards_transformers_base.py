# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    list_comp_0 = module_0.ListComp()
    base_node_transformer_0 = module_1.BaseNodeTransformer(list_comp_0)


def test_case_1():
    call_0 = module_0.Call()
    base_import_rewrite_0 = module_1.BaseImportRewrite(call_0)
    list_0 = [call_0, base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    call_0 = module_0.Call()
    base_import_rewrite_0 = module_1.BaseImportRewrite(call_0)
    none_type_0 = None
    list_0 = [none_type_0, none_type_0, none_type_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    call_0 = module_0.Call()
    var_0 = module_2.iter_fields(call_0)
    list_0 = [call_0, var_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    var_1 = base_import_rewrite_0.visit(import_from_0)


def test_case_4():
    call_0 = module_0.Call()
    var_0 = module_2.dump(call_0)
    list_0 = [call_0, var_0, var_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0 = module_1.BaseImportRewrite(import_from_0)
    base_import_rewrite_0.visit(import_from_0)
