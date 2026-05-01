
class Test{
public:
    Test() = default;

    ~Test();

    Test(const Test& other) = default;
    Test(Test&& other) = default;

    Test& operator=(const Test& other) = default;
    Test& operator=(Test&& other) = default;
    
    
}