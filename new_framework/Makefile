CXX = icc #g++
MPICC = mpic++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 #-march=haswell
ADDITIONAL_FLAG = -D_DEBUG   #####debug purpose...-D_HOGWILD_SHORT  -DAVX2_EN -DCPU_BINDING_EN
#ADDITIONAL_FLAG = -D_PRINT_GRADIENT
#ADDITIONAL_FLAG = -DLOAD_FILE_PER_WORKER -D_DEBUG
CPP_INCLUDE = -I./BitWeaving -I./hazytl/include/ -I../IntelPCM/ -I./linearmodel/ -I. -I./tool/
CPP_LAST = -lpthread

BIN_DIR = bin
APP_DIR = app

TEST_AVX2 = ${BIN_DIR}/TEST_AVX #${BIN_DIR}/TEST_FVECTOR

#SRC_FILES = src/executor.h
SRC_FILES += linearmodel/linearmodel.h 

#SYNC ASYNC ALLREDUCE ASYNCSPARSE ALLREDUCEBINOMIAL ALLREDUCESPARSEBIG ALLREDUCESPARSESMALL ALLREDUCESPARSERECDBL
STRATEGY := MODELSYNC #HOGWILD 
STEPSIZE := NON  # DECREASING_STEPSIZES EXPBACKOFF_STEPSIZES
REPRESENTATION := DENSE # SPARSE
SIMD := SCALAR AVX 
FORMAT:= INT #SHORT FP CHAR 
THREADBING:= BIND RANDOM

.PHONY: depend clean all
all: LINREG ${TEST_AVX2}

ALLEXECS = $(LINREGS)
#LOGIT SVM #$(SPLIT_DATA_TOOL) ${RANDOM_RESHUFFLE_TOOL} ${GENERATE_SPLIT_FILE_TOOL}

clean: CLEAN-LINREG 
	rm -f ${TEST_AVX2}

LINREGS := $(foreach SY,$(STRATEGY), $(foreach SE, $(STEPSIZE), $(foreach RN, $(REPRESENTATION), $(foreach FM, $(FORMAT), $(foreach SD, $(SIMD), $(foreach TB, $(THREADBING), $(BIN_DIR)/LINREG_$(SY)_$(SE)_$(RN)_$(SD)_$(FM)_$(TB) ) ) ) ) ) )

CLEAN-LINREG:
	rm -f $(LINREGS)


$(LINREGS): ${APP_DIR}/linreg/linreg.cpp $(SRC_FILES)
	mkdir -p $(BIN_DIR)
	${CXX} $(CPP_FLAG) ${ADDITIONAL_FLAG} $(PARAMS) $(CPP_INCLUDE) $< -o $@ $(CPP_LAST) ../IntelPCM/libpcm_2_11.a

LINREG: $(LINREGS)


# Add strategy specific options
$(foreach f, $(ALLEXECS), $(if $(findstring _HOGWILD_, $f),$f,)): strategy/hogwild.h
$(foreach f, $(ALLEXECS), $(if $(findstring _MODELSYNC_, $f),$f,)): strategy/modelsync.h

# Add representation
$(foreach f, $(ALLEXECS), $(if $(findstring _DENSE_, $f),$f,)): PARAMS += -D_DENSE
$(foreach f, $(ALLEXECS), $(if $(findstring _SPARSE_, $f),$f,)):PARAMS += -D_SPARSE
$(foreach f, $(ALLEXECS), $(if $(findstring _SHORT, $f),$f,)):PARAMS += -D_HOGWILD_SHORT
$(foreach f, $(ALLEXECS), $(if $(findstring _CHAR,  $f),$f,)):PARAMS += -D_HOGWILD_CHAR
$(foreach f, $(ALLEXECS), $(if $(findstring _INT,  $f),$f,)):PARAMS += -D_HOGWILD_INT

$(foreach f, $(ALLEXECS), $(if $(findstring _MODELSYNC,  $f),$f,)):PARAMS += -D_MODELSYNC


$(foreach f, $(ALLEXECS), $(if $(findstring _AVX,  $f),$f,)):PARAMS   += -DAVX2_EN
$(foreach f, $(ALLEXECS), $(if $(findstring _BIND,  $f),$f,)):PARAMS  += -DCPU_BINDING_EN


# Add Stepsize behavior

$(filter %_DECREASING_STEPSIZES_DENSE %_DECREASING_STEPSIZES_SPARSE,$(ALLEXECS)): PARAMS += -D_DECREASING_STEPSIZES
$(filter %_EXPBACKOFF_STEPSIZES_DENSE %_EXPBACKOFF_STEPSIZES_SPARSE,$(ALLEXECS)): PARAMS += -D_EXPBACKOFF_STEPSIZES

# Add representation

#$(filter %_DENSE,$(ALLEXECS)): PARAMS += -D_DENSE
#$(filter %_SPARSE,$(ALLEXECS)): PARAMS += -D_SPARSE

