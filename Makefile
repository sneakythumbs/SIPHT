#export LD_LIBRARY_PATH= '/home/kenny/Courses/Semester4/OpenCV/myOpenCV'
#export CPLUS_INCLUDE_PATH='/home/kenny/Courses/Semester4/OpenCV/myOpenCV/include'
CXXFLAGS+= -std=c++11 -O2 -pipe -g -I '/home/kenny/Courses/Semester4/OpenCV/myOpenCV/include'
#CXXFLAGS+=-O2 -pipe 
#CXXFLAGS+=-O0 -Wall -M -pipe -I /home/kenny/Courses/Semester4/OpenCV/myOpenCV

LDADD=	-L${HOME}/Courses/Semester4/OpenCV/myOpenCV/lib \
	-lopencv_core -lopencv_highgui -lopencv_features2d \
	-lopencv_imgproc -lopencv_flann

%.o: %.cpp %.hpp
	${CXX} -c ${CXXFLAGS} $<

skewer: pk.o sipht.o skewer.o 
	${CXX} $^ -o $@ ${LDADD} #-Wl,--trace
	
test: test.o sipht.o
	${CXX} $^ -o $@	${LDADD}
	
harris: pk.o Harris_Laplace.o Harris_Test.o
	${CXX} $^ -o $@	${LDADD}
	
laplace: sipht.o Laplace.o Laplace_Test.o
	${CXX} $^ -o $@	${LDADD}
	
hessian: pk.o sipht.o Hessian_Laplace.o Hessian_Test.o
	${CXX} $^ -o $@	${LDADD}
	
mser: pk.o MSER_Test.o
	${CXX} $^ -o $@	${LDADD}

orb: pk.o sipht.o Orb_Test.o
	${CXX} $^ -o $@	${LDADD}
	
fast: pk.o FAST_Laplace.o FAST_Test.o
	${CXX} $^ -o $@	${LDADD}
	
fastharris: pk.o FAST_Harris.o FAST_Harris_Test.o
	${CXX} $^ -o $@	${LDADD}
