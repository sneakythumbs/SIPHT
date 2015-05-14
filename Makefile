#export LD_LIBRARY_PATH= '/home/kenny/Courses/Semester4/OpenCV/myOpenCV'
#export CPLUS_INCLUDE_PATH='/home/kenny/Courses/Semester4/OpenCV/myOpenCV/include'
CXXFLAGS+= -std=c++11 -O2 -pipe -g -I '/home/kenny/Courses/Semester4/OpenCV/myOpenCV/include'
#CXXFLAGS+=-O2 -pipe 
#CXXFLAGS+=-O0 -Wall -M -pipe -I /home/kenny/Courses/Semester4/OpenCV/myOpenCV

LDADD=	-L${HOME}/Courses/Semester4/OpenCV/myOpenCV/lib \
	-lopencv_core -lopencv_highgui -lopencv_features2d \
	-lopencv_imgproc -lopencv_flann

%.o: %.cpp
	${CXX} -c ${CXXFLAGS} $<

skewer: skewer.o sipht.o
	${CXX} $^ -o $@ ${LDADD} #-Wl,--trace