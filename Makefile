build:
	nvcc src/main.cu -o bin/project.exe `pkg-config opencv4 --cflags --libs`

clean:
	rm -f bin/project.exe