# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import py_backwards.transformers.base as module_0
import typed_ast._ast3 as module_1
import typed_ast.ast3 as module_2


def test_case_0():
    none_type_0 = None
    base_node_transformer_0 = module_0.BaseNodeTransformer(none_type_0)


def test_case_1():
    lt_0 = module_1.Lt()
    none_type_0 = None
    base_import_rewrite_0 = module_0.BaseImportRewrite(lt_0)
    bytes_0 = module_1.Bytes()
    list_0 = [none_type_0, bytes_0, bytes_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    lt_0 = module_1.Lt()
    base_import_rewrite_0 = module_0.BaseImportRewrite(lt_0)
    list_0 = [base_import_rewrite_0, base_import_rewrite_0, base_import_rewrite_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    lt_0 = module_1.Lt()
    base_import_rewrite_0 = module_0.BaseImportRewrite(lt_0)
    var_0 = module_2.iter_child_nodes(lt_0)
    list_0 = [var_0, var_0, var_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    var_1 = base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    lt_0 = module_1.Lt()
    base_import_rewrite_0 = module_0.BaseImportRewrite(lt_0)
    var_0 = module_2.walk(lt_0)
    list_0 = [var_0, var_0, var_0]
    import_from_0 = module_1.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)
