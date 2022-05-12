TARGET := main

SOURCES := main.c ./src/*.c
CFLAGS := -I./include

$(TARGET): $(SOURCES)
	@$(CC) $(CFLAGS) -g -o $@ $(SOURCES) -lm

clean:
	rm -f *.o $(TARGET)