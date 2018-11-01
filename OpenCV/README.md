Run the OpenCV files with the following command:

g++ main.cpp -o output `pkg-config --cflags --libs opencv`

Where main.cpp is the file name, output is the executable name

Generic run command

g++ <fileName>.cpp -o <executable_output> `pkg-config --cflags --libs opencv`
