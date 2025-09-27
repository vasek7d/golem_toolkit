from golem_toolkit.test_package.hello import hello

def test_hello():
    assert hello() == "Hello from test_package!"