.SUFFIXES: .c .o .f .F

CC			=  mpicc  
CFLAGS			=  -g -Wall

FILES =  main.o aux.o hqsort.o
StatFiles = mainStats.o aux.o hqsort.o

main.ex: $(FILES) 
	${CC} ${CFLAGS} -o main.ex -lm $(FILES) 

mainStats.ex: $(StatFiles)
	${CC} ${CFLAGS} -o mainStats.ex -lm $(StatFiles)

.c.o:
	${CC} ${CFLAGS} $< -c -o $@

clean:
	rm *.o *.ex
