/*model*/

/*HM*/
var HM = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('HM');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var CtorHM = Vue.extend(HM)
new CtorHM().$mount('#HM')

/*HMM*/
var HMM = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('HMM');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var CtorHMM = Vue.extend(HMM)
new CtorHMM().$mount('#HMM')

/*ARIMA*/
var ARIMA = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('ARIMA');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor5 = Vue.extend(ARIMA)
new Ctor5().$mount('#ARIMA')

/*DCRNN*/
var DCRNN = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('DCRNN');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor6 = Vue.extend(DCRNN)
new Ctor6().$mount('#DCRNN')

/*DeepST*/
/*var DeepST = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('DeepST');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor7 = Vue.extend(DeepST)
new Ctor7().$mount('#DeepST')*/

/*GeoMAN*/
var GeoMAN = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('GeoMAN');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor8 = Vue.extend(GeoMAN)
new Ctor8().$mount('#GeoMAN')

/*ST-MGCN*/
var ST_MGCN = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('ST-MGCN');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor9 = Vue.extend(ST_MGCN)
new Ctor9().$mount('#ST_MGCN')

/*ST-ResNet*/
/*var ST_ResNet = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('ST_ResNet');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor10 = Vue.extend(ST_ResNet)
new Ctor10().$mount('#ST_ResNet')*/

/*XGBoost*/
var XGBoost = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('XGBoost');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor11 = Vue.extend(XGBoost)
new Ctor11().$mount('#XGBoost')

/*GBRT*/
var GBRT = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('GBRT');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var CtorGBRT = Vue.extend(GBRT)
new CtorGBRT().$mount('#GBRT')

/*STMeta*/
/*var STMeta = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('STMeta');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor12 = Vue.extend(STMeta)
new Ctor12().$mount('#STMeta')*/

/*Test*/
/*var TMeta = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('TMeta_C6P0T0_GD_K0L1_F60_LSTMC');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor_test1 = Vue.extend(TMeta)
new Ctor_test1().$mount('#TMeta_C6P0T0_GD_K0L1_F60_LSTMC')*/
/*var V1 = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('V1_C6P7T4_GDCL_K1L1_F60_BM');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor_V1 = Vue.extend(V1)
new Ctor_V1().$mount('#V1_C6P7T4_GDCL_K1L1_F60_BM')*/
/*
var V2 = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('V2_C6P7T4_GDCL_K1L1_F60_BM');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor_V2 = Vue.extend(V2)
new Ctor_V2().$mount('#V2_C6P7T4_GDCL_K1L1_F60_BM')*/
/*
var V3 = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            var methodid = MethodNameArray.map(item => item).indexOf('V3_C6P7T4_GDCL_K1L1_F60_BM');
            if(methodid < 0 || methodid > (FunctionNum-2) ){
                alert("The model does not exist!")
            }
            ChangeMethod(methodid);
        }
    }
}
var Ctor_V3 = Vue.extend(V3)
new Ctor_V3().$mount('#V3_C6P7T4_GDCL_K1L1_F60_BM')*/
