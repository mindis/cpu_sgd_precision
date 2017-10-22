/*
Copyright (c) 2009-2012, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// written by Roman Dementiev
//            Austen Ott

#include <iostream>
#include <assert.h>
#include <unistd.h>
#include "msr.h"

#include "perf_counters.h"


#define NUM_CORES 16 


struct Monitor_Event inst_Monitor_Event = {
	{
		{0x2e,0x41},
		{0xd2,0x08},
		{0xd2,0x07},
		{0xd1,0x02},
	},
	1,
	{
		"UOPS_ISSUED.ANY: ",
		"DTLB_LOAD_MISSES: ",
		"OPS_ISSUED: ",
		"DTLB_LOAD_MISSES: ",
	},
	{
		{0,0},
		{0,0},
		{0,0},
		{0,0},		
	},
	2,
	{
		"MIC_0",
		"MIC_1",
		"MIC_2",
		"MIC_3",
	},
    0	 
};



int main()
{
    uint32 i = 0;
    uint32 res;
    MsrHandle * cpu_msr[NUM_CORES];


	PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
	PCM_start();


	PCM_stop();
	printf("=====print the profiling result==========\n");

	PCM_printResults(); 	
	PCM_cleanup();


}
