
.SUFFIXES: .c .o .f .F

CC			=  gcc 
CFLAGS			= -Wall -O3 -fopenmp 

FILES =  matmat.o timer.o 
LIB = -llapack -lblas

test.ex: $(FILES) 
	${CC} ${CFLAGS} -o test.ex $(FILES) $(LIB)

.c.o:
	${CC} ${CFLAGS} $< -c -o $@ $(LIB)

clean:
	rm *.o *.ex
