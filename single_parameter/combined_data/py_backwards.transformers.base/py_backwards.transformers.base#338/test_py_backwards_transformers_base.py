# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import typed_ast._ast3 as module_0
import py_backwards.transformers.base as module_1


def test_case_0():
    add_0 = module_0.Add()
    base_node_transformer_0 = module_1.BaseNodeTransformer(add_0)


def test_case_1():
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [base_import_rewrite_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_2():
    mod_0 = module_0.Mod()
    none_type_0 = None
    base_import_rewrite_0 = module_1.BaseImportRewrite(none_type_0)
    list_0 = [none_type_0, none_type_0, mod_0]
    str_0 = "\x0bdMV"
    dict_0 = {str_0: str_0, str_0: base_import_rewrite_0, str_0: mod_0, str_0: str_0}
    import_from_0 = module_0.ImportFrom(*list_0, **dict_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_3():
    mod_0 = module_0.Mod()
    str_0 = "C<uzlT;JAn1 JMNRS"
    base_import_rewrite_0 = module_1.BaseImportRewrite(mod_0)
    list_0 = [mod_0, str_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    base_import_rewrite_0.visit_ImportFrom(import_from_0)


def test_case_4():
    mod_0 = module_0.Mod()
    str_0 = ""
    base_import_rewrite_0 = module_1.BaseImportRewrite(mod_0)
    list_0 = [mod_0, str_0]
    import_from_0 = module_0.ImportFrom(*list_0)
    var_0 = base_import_rewrite_0.visit_ImportFrom(import_from_0)
