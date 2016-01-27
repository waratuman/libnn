uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')

# augment via OPTFLAGS
CFLAGS=-g -O2 -Wall -Wextra -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -Isrc -rdynamic -Wstrict-overflow -fno-strict-aliasing -DNDEBUG $(OPTFLAGS)

# augment linking options via OPTLIBS
LDLIBS=-ldl -lm -lblas $(OPTLIBS)

# CBLAS
ifeq ($(uname_S),Darwin)
	# LDFLAGS += -L/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current
	# LDLIBS += -l/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current
	CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers
endif

# optional, only applies when user didn't give PREFIX setting
PREFIX?=/usr/local

# all .c files in source and below
SOURCES=$(wildcard src/**/*.c src/*.c)
HEADERS=$(wildcard src/**/*.h src/*.h)
OBJECTS=$(SOURCES:.c=.o)

TEST_SRC=$(wildcard tests/**/*.c tests/*.c)
TESTS=$(TEST_SRC:.c=)

TARGET=build/libnn.a
SO_TARGET=$(TARGET:.a=.so)

all: $(TARGET) $(SO_TARGET)

# changing options for just the developer build (-Wextra is useful for finding bugs)
dev: CFLAGS=-g -Wall -Isrc -Wall -Wextra $(OPTFLAGS)
dev: all

# adding -fPIC option for just the target build
$(TARGET): CFLAGS += -fPIC
# makes the target, first the .a file (ar) and then the library via ranlib
$(TARGET): build $(OBJECTS)
	ar rcs $@ $(OBJECTS)
	ranlib $@

$(SO_TARGET): $(TARGET) $(OBJECTS)
	$(CC) -shared -o $@ $(OBJECTS) $(LDLIBS)

build:
	@mkdir -p build
	@mkdir -p bin

# link the test programs with the TARGET library, i.e. build/libYOUR_LIBRARY
test: CFLAGS += -Itests $(TARGET)
test: $(TESTS)
	sh ./tests/runtests.sh

clean:
	rm -rf build $(OBJECTS) $(TESTS)
	rm -f tests/tests.log
	find . -name "*.gc*" -exec rm {} \;
	rm -rf `find . -name "*.dSYM" -print`

install: all
ifeq ($(uname_S),Darwin)
	install -d $(DESTDIR)/$(PREFIX)/lib/
	install -d $(DESTDIR)/$(PREFIX)/include/nn/
	install -m755 $(TARGET) $(DESTDIR)/$(PREFIX)/lib/
	install -m644 $(HEADERS) $(DESTDIR)/$(PREFIX)/include/nn/
else # ifeq ($(uname_S),Linux)
	install -Dm755 $(TARGET) $(DESTDIR)/$(PREFIX)/lib/
	install -Dm644 $(HEADERS) $(DESTDIR)/$(PREFIX)/include/nn/
endif

BADFUNCS='[^_.>a-zA-Z0-9](str(n?cpy|n?cat|xfrm|n?dup|str|pbrk|tok|_)|stpn?cpy|a?sn?printf|byte_)'
check:
	@echo Files with potentially dangerous functions.
	@egrep $(BADFUNCS) $(SOURCES) || true

.PHONY: all dev build tests 