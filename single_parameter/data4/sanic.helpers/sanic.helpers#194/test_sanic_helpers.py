# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import sanic.helpers as module_0


def test_case_0():
    int_0 = 200
    var_0 = module_0.has_message_body(int_0)


def test_case_1():
    str_0 = "Retrieve a `Route` object containing the details about how to handle a response for a given request\n\n        :param request: the incoming request object\n        :type request: Request\n        :return: details needed for handling the request and returning the\n            correct response\n        :rtype: Tuple[ Route, RouteHandler, Dict[str, Any]]\n\n        Args:\n            path (str): the path of the route\n            method (str): the HTTP method of the route\n            host (Optional[str]): the host of the route\n\n        Raises:\n            NotFound: if the route is not found\n            MethodNotAllowed: if the method is not allowed for the route\n\n        Returns:\n            Tuple[Route, RouteHandler, Dict[str, Any]]: the route, handler, and match info\n        "
    dict_0 = {str_0: str_0}
    var_0 = module_0.remove_entity_headers(dict_0)


def test_case_2():
    bool_0 = module_0.is_atty()


def test_case_3():
    none_type_0 = None
    module_0.is_hop_by_hop_header(none_type_0)


def test_case_4():
    default_0 = module_0.Default()
    var_0 = default_0.__repr__()
    int_0 = 200
    var_1 = module_0.has_message_body(int_0)
    bool_0 = module_0.is_atty()


def test_case_5():
    float_0 = 82.0
    var_0 = module_0.has_message_body(float_0)
    bool_0 = module_0.is_atty()
    default_0 = module_0.Default()
    str_0 = default_0.__str__()
    module_0.remove_entity_headers(float_0)


def test_case_6():
    bool_0 = True
    var_0 = module_0.has_message_body(bool_0)
    var_1 = var_0.__repr__()
    module_0.Default(**bool_0)


def test_case_7():
    int_0 = 204
    var_0 = module_0.has_message_body(int_0)
