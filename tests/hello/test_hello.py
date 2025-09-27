from golem_toolkit.hello_package.hello import hello

def test_hello():
    assert hello() == "Hello from test_package!"