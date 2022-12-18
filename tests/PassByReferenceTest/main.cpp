#include <iostream>
#include <vector>

using namespace std;

vector<int> test(const vector<int>& inp) {
    vector<int> empty;
    inp = empty;

    return inp;
}

void printVector(vector<int>& inp) {
    for (int a : inp) {
        cout << a << " ";
    }

    cout << endl;
}

int main() {
    vector<int> x;
    x.push_back(1);
    x.push_back(2);
    x.push_back(3);

    printVector(x);
    vector<int> y = test(x);

    printVector(x);
    printVector(y);
}