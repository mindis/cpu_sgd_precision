CXX = icc #g++
MPICC = mpic++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 #-march=haswell
ADDITIONAL_FLAG = -D_DEBUG   #####debug purpose...-D_HOGWILD_SHORT 
#ADDITIONAL_FLAG = -D_PRINT_GRADIENT
#ADDITIONAL_FLAG = -DLOAD_FILE_PER_WORKER -D_DEBUG
CPP_INCLUDE = -I./src -I./hazytl/include/
CPP_LAST = -lpthread

BIN_DIR = bin
APP_DIR = app
SPLIT_DATA_TOOL = ${BIN_DIR}/tools/split_data.out
RANDOM_RESHUFFLE_TOOL = ${BIN_DIR}/tools/random_reshuffle.out
GENERATE_SPLIT_FILE_TOOL = ${BIN_DIR}/tools/generate_split_file.out
SRC_FILES = src/executor.h
SRC_FILES += app/linearmodel_exec.h app/linearmodel/linearmodel.h app/linearmodel/linearmodel_loader.h

#SYNC ASYNC ALLREDUCE ASYNCSPARSE ALLREDUCEBINOMIAL ALLREDUCESPARSEBIG ALLREDUCESPARSESMALL ALLREDUCESPARSERECDBL
STRATEGY := HOGWILD 
STEPSIZE := NON  # DECREASING_STEPSIZES EXPBACKOFF_STEPSIZES
REPRESENTATION := DENSE # SPARSE
SIMD := SCALAR AVX 
FORMAT:= SHORT FP #CHAR 

.PHONY: depend clean all
all: LINREG 
#LOGIT SVM #$(SPLIT_DATA_TOOL) ${RANDOM_RESHUFFLE_TOOL} ${GENERATE_SPLIT_FILE_TOOL}

clean: CLEAN-LINREG CLEAN-LOGIT CLEAN-SVM
	rm -f ${BIN_DIR}/tools/*.out

LINREGS := $(foreach SY,$(STRATEGY), $(foreach SE, $(STEPSIZE), $(foreach RN, $(REPRESENTATION), $(foreach FM, $(FORMAT), $(foreach SD, $(SIMD), $(BIN_DIR)/LINREG_$(SY)_$(SE)_$(RN)_$(SD)_$(FM) ) ) ) ) )
LOGITS  := $(foreach SY,$(STRATEGY), $(foreach SE, $(STEPSIZE), $(foreach RN, $(REPRESENTATION), $(foreach FM, $(FORMAT), $(foreach SD, $(SIMD),  $(BIN_DIR)/LOGIT_$(SY)_$(SE)_$(RN)_$(SD)_$(FM) ) ) ) ) )
SVMS    := $(foreach SY,$(STRATEGY), $(foreach SE, $(STEPSIZE), $(foreach RN, $(REPRESENTATION), $(foreach FM, $(FORMAT), $(foreach SD, $(SIMD),    $(BIN_DIR)/SVM_$(SY)_$(SE)_$(RN)_$(SD)_$(FM) ) ) ) ) )

ALLEXECS = $(LINREGS) $(LOGITS) $(SVMS)

CLEAN-LINREG:
	rm -f $(LINREGS)

CLEAN-LOGIT:
	rm -f $(LOGITS)

CLEAN-SVM:
	rm -f $(SVMS)

$(LINREGS): ${APP_DIR}/linreg.cpp ${APP_DIR}/linreg/linreg_exec.h $(SRC_FILES)
	mkdir -p $(BIN_DIR)
	${CXX} $(CPP_FLAG) ${ADDITIONAL_FLAG} $(PARAMS) $(CPP_INCLUDE) $< -o $@ $(CPP_LAST) 

$(LOGITS): ${APP_DIR}/logit.cpp ${APP_DIR}/logit/logit_exec.h $(SRC_FILES)
	mkdir -p $(BIN_DIR)
	${CXX} $(CPP_FLAG) ${ADDITIONAL_FLAG} $(PARAMS) $(CPP_INCLUDE) $< -o $@ $(CPP_LAST) 

$(SVMS): ${APP_DIR}/svm.cpp ${APP_DIR}/svm/svm_exec.h $(SRC_FILES)
	mkdir -p $(BIN_DIR)
	${CXX} $(CPP_FLAG) ${ADDITIONAL_FLAG} $(PARAMS) $(CPP_INCLUDE) $< -o $@ $(CPP_LAST) 

LINREG: $(LINREGS)
LOGIT: $(LOGITS)
SVM: $(SVMS)

# Add strategy specific options
$(foreach f, $(ALLEXECS), $(if $(findstring _HOGWILD_, $f),$f,)): src/strategy/hogwild.h

# Add representation
$(foreach f, $(ALLEXECS), $(if $(findstring _DENSE_, $f),$f,)): PARAMS += -D_DENSE
$(foreach f, $(ALLEXECS), $(if $(findstring _SPARSE_, $f),$f,)):PARAMS += -D_SPARSE
$(foreach f, $(ALLEXECS), $(if $(findstring _SHORT, $f),$f,)):PARAMS += -D_HOGWILD_SHORT
$(foreach f, $(ALLEXECS), $(if $(findstring _CHAR,  $f),$f,)):PARAMS += -D_HOGWILD_CHAR


# Add Stepsize behavior

$(filter %_DECREASING_STEPSIZES_DENSE %_DECREASING_STEPSIZES_SPARSE,$(ALLEXECS)): PARAMS += -D_DECREASING_STEPSIZES
$(filter %_EXPBACKOFF_STEPSIZES_DENSE %_EXPBACKOFF_STEPSIZES_SPARSE,$(ALLEXECS)): PARAMS += -D_EXPBACKOFF_STEPSIZES

# Add representation

#$(filter %_DENSE,$(ALLEXECS)): PARAMS += -D_DENSE
#$(filter %_SPARSE,$(ALLEXECS)): PARAMS += -D_SPARSE

# Add simd
$(filter %_AVX,$(ALLEXECS)): PARAMS += -DAVX2_EN



${SPLIT_DATA_TOOL}: src/tools/split_data.cpp
	mkdir -p $(BIN_DIR)/tools/
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

${RANDOM_RESHUFFLE_TOOL}: src/tools/random_reshuffle.cpp
	mkdir -p $(BIN_DIR)/tools/
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

${GENERATE_SPLIT_FILE_TOOL}: src/tools/generate_split_file.cpp
	mkdir -p $(BIN_DIR)/tools/
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@
