// datasets name
var data = "";                               //选定数据集后把数据转到这里，方便统一处理
var Metro_Chongqing = "";
var DiDi_Xian = "";
var Violation_XM = "";
var Bike_NYC = "";
var Bike_DC = "";
var Bike_Chicago = "";

// datasets basic info
var TimeArrayLength = 0;                     // 标记当前选择的数据集中，每个数据点，其对应的时间节点有多少个
var VaildPointNum = 0;                       // 标记当前选择的数据集中，GroundTruth中包含多少个地图节点的全局变量(实际是有效节点量)
var TotalPointNum = 0;                       // 标记StationInfo里头有标记的所有节点的数量
var TimeRange = [];                          // 标记当前选择的数据集的TimeRange，第三个元素为结束日期的后一天
var now_Timefitness = 0;                     // 存储当前选中数据集的timefitness

// children datasets param
var portion_index = 1;                       // 数据集第二位会有一个参数，表明使用了该数据集的多少。如果为all的话，就是全部使用
var datatype_index = 'N';                    // 数据集8个参数中的最后一个参数，表明是节点型还是网格型数据。默认为N（节点型）
var DatasetID = 0;                           // 单个json文件中包含多个数据集，这里用来记录使用的是哪号数据集。
var DatasetList = new Array();               // 存储数据集名称的列表

// model(methods)
var MethodNameArray = new Array();           // 记录方法的名称（不包含GroundTruth）
var MethodID = 0;                            // 用来记录使用的是哪种方法（不包含GroundTruth）所以初始化时默认展示的是第一种方法
var FunctionNum = 0;                         // 标记数据集中含有的方法数(包含了GroundTruth)

// evaluation metrics
var PointVar = new Array();                  // 二维数组,(FunctionNum,VaildPointNum)数量级，表示对此数据集，每个方法对应每个地图点的RMSE
var PointSortedVar = new Array();            // 二维数组，降序排列的RMSE
var PointMinVar = new Array();               // 一维数组，代表每个点多种方法中的最小的RMSE
var PointMAE = new Array();                  // 二维数组，存每个节点的MAE值（仅显示，暂不处理）

// map point
var pointID = 0;                             // 当前用来做图的那个地图点，以及时间上的起止
var pointName = ""
var StartInd = -1;
var EndInd = -1;
var myMapSize = 13;                          // 默认地图尺寸
var line_highlight = 0;                      // 折线图的点是否用颜色表示相对误差的值，默认不开启此功能
var MAPInd = 0;                              // 选择哪个地图

var MAXError = 0.1;       // 单点认为判断准确的最大相对误差
var MAXABSError = 1       // 单点认为判断准确的最大绝对误差
var MINACCURACY = 1;     // 认为某方法能准确预测地图点的流量的最低准确率 RMSE

var MODE = 0;               // 标记位，表示不一样的方法。0表示使用相对误差，1表示绝对误差

// model error
var rmse = '';
var mape = '';
var mae = '';


