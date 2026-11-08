#pragma once
// CPU effect (After Effects API) headers.

#include "AEConfig.h"
#ifdef AE_OS_WIN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h" // PF_ParamUtilsSuite3 (PF_UpdateParamUI)
#include "Param_Utils.h"
#include "SPBasic.h" // defines SPAPI, needed by PrSDKAESupport.h
#include "PrSDKAESupport.h"

#include "OneEightyParams.h"

extern "C"
{
    DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                PF_LayerDef* output, void* extra);
}
