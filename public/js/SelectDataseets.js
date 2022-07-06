/*选择数据集*/

/*NYC*/
var NYC = {
    methods: {
        test() {

            data = Bike_NYC;
            MAPInd = 0; // 选择地图

            // 获得该数据集下各个子数据集
            ClearDataSet();
            for (x in data.Pred) {
                DatasetList.push(x);
            }
            StartDataSet();

        }
    }
}
var Ctor2 = Vue.extend(NYC)
new Ctor2().$mount('#NYC')

/*DC*/
var DC = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            data = Bike_DC;
            MAPInd = 0; // 选择地图

            // 获得该数据集下各个子数据集
            ClearDataSet();
            for (x in data.Pred) {
                DatasetList.push(x);
            }
            StartDataSet();
        }
    }
}
var Ctor3 = Vue.extend(DC)
new Ctor3().$mount('#DC')

/*Chicago*/
var Chicago = {
    data () {
        return {
            value: true
        };
    },
    methods: {
        test() {
            data = Bike_Chicago;
            MAPInd = 0; // 选择地图

            // 获得该数据集下各个子数据集
            ClearDataSet();
            for (x in data.Pred) {
                DatasetList.push(x);
            }
            StartDataSet();
        }
    }
}
var Ctor4 = Vue.extend(Chicago)
new Ctor4().$mount('#Chicago')

/*New*/
// var New = {
//     data () {
//         return {
//             value: true
//         };
//     },
//     methods: {
//         test() {
//             console.log("Please upload dataset file!")
//             // data = Bike_Chicago;
//             // MAPInd = 0; // 选择地图
//
//             // 获得该数据集下各个子数据集
//             // ClearDataSet();
//             // for (x in data.Pred) {
//             //     DatasetList.push(x);
//             // }
//             // StartDataSet();
//         }
//     }
// }
// var Ctor5 = Vue.extend(New)
// new Ctor5().$mount('#New')


/*Chongqing*/
/*var Chongqing = {
    methods: {
        test() {

            data = Metro_Chongqing;
            MAPInd = 0;

            // 获得该数据集下各个子数据集
            ClearDataSet();
            for (x in data.Pred) {
                DatasetList.push(x);
            }

            StartDataSet();

        }
    }
}
var Ctor_Chongqing = Vue.extend(Chongqing)
new Ctor_Chongqing().$mount('#Chongqing')*/

/*Xian*/
/*var Xian = {
    methods: {
        test() {

            data = DiDi_Xian;
            MAPInd = 0;

            // 获得该数据集下各个子数据集
            ClearDataSet();
            for (x in data.Pred) {
                DatasetList.push(x);
            }

            StartDataSet();
        }
    }
}
var Ctor_Xian = Vue.extend(Xian)
new Ctor_Xian().$mount('#Xian')*/

/*Xiamen*/
var Xiamen = {
    methods: {
        test() {

            data = Violation_XM;
            MAPInd = 0; // 只有结点的地图
            // MAPInd = 1; // 有区域的地图

            // 获得该数据集下各个子数据集
            ClearDataSet();
            DatasetID = 0; // 让厦门数据集默认展示674的子数据集
            for (x in data.Pred) {
                DatasetList.push(x);
            }

            StartDataSet();
        }
    }
}
var Ctor_Xiamen = Vue.extend(Xiamen)
new Ctor_Xiamen().$mount('#Xiamen')