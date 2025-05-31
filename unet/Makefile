CUXX    = nvcc
CUFLAGS = --compiler-options -Wall --compiler-options -Wextra -Xcompiler -fPIC -std=c++14 --expt-relaxed-constexpr -I./include

ifdef DEBUG
CUFLAGS += -g -G
else
CUFLAGS += -O3 --use_fast_math
endif

SRC := $(shell find ./src -name '*.cu')
OBJ := $(SRC:.cu=.o)

TARGET = libunet.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CUXX) $(CUFLAGS) -shared -o $@ $^

./src/%.o: ./src/%.cu
	$(CUXX) $(CUFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
