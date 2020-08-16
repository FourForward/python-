[TOC]
# 一、 JavaScript 概述
 ## 1. 什么是JavaScript
#### 1) JS 介绍
简称JS，是一种浏览器解释型语言,嵌套在HTML文件中交给浏览器解释执行。主要用来实现网页的动态效果，用户交互及前后端的数据传输等。
#### 2) JS 组成
1. 核心语法 -ECMAScript 规范了JS的基本语法
2. 浏览器对象模型 -BOM
     Browser Object Model，提供了一系列操作浏览器的方法
3. 文档对象模型 -DOM
     Document Object Model ，提供了一系列操作的文档的方法
## 2. 使用方式
1. 元素绑定事件
      + 事件 ：指用户的行为（单击，双击等）或元素的状态（输入框的焦点状态等）
      + 事件处理：元素监听某种事件并在事件发生后自动执行事件处理函数。
      + 常用事件：onclick (单击事件) 
      + 语法 ：将事件名称以标签属性的方式绑定到元素上，自定义事件处理。
      ```html
      <!--实现点击按钮在控制台输出-->
      <button onclick="console.log('Hello World');">点击</button>
      ```
2. 文档内嵌。使用<script type="text/javascript"></script>标签书写 JS 代码
      + 语法 ：
      ```html
      <script type="text/javascript">
        alert("网页警告框");
      </script>
      ```
      + 注意 ：<script></script>标签可以书写在文档的任意位置，书写多次，一旦加载到script标签就会立即执行内部的JS代码，因此不同的位置会影响代码最终的执行效果
3. 外部链接
      + 创建外部的JS文件 XX.js，在HTML文档中使用<script src=""></script>引入
      ```html
      <script src="index.js"></script>
      ```
      + 注意 ：<script></script>既可以实现内嵌 JS 代码，也可以实现引入外部的 JS 文件，但是只能二选一。

# 二、基础语法

## 1. 语法规范
1. JS是由语句组成,语句由关键字,变量,常量,运算符,方法组成.分号可以作为语句结束的标志,也可以省略
2. JS严格区分大小写
3. 注释语法
    单行注释使用 //
    多行注释使用 /* */
## 2. JS的变量与常量
#### 1)  变量
1. 作用 : 用于存储程序运行过程中可动态修改的数据
2. 语法 : 使用关键var声明,自定义变量名
    ```javascript
    var a;		//变量声明
    a = 100;	//变量赋值
    var b = 200; //声明并赋值
    var m,n,k;	//同时声明多个变量
    var j = 10,c = 20; //同时声明并赋值多个变量
    let a = 1;	//申明局部作用域变量，不会发生申明提前，适合在函数或者循环中使用
    // 不加var申明的变量，是全局变量
    //【申明提前】
    	变量申明提前：会将变量名提到最前方，值留在原地
        function申明提前：会携带函数体提到前方
    ```
3. 命名规范 : 
      + 变量名,常量名,函数名,方法名自定义,可以由数字,字母,下划线,$组成,禁止以数字开头
      + 禁止与关键字冲突(var const function if else for while do break case switch return class)
      + 变量名严格区分大小写
      + 变量名尽量见名知意,多个单词组成采用小驼峰,例如："userName"
4. 使用注意 :
      + 变量如果省略var关键字,并且未赋值,直接访问会报错
      + 变量使用var关键字声明但未赋值,变量初始值为undefined
      + 变量省略var关键字声明,已被赋值,可正常使用.影响变量作用域
#### 2)  常量 
1. 作用 : 存储一经定义就无法修改的数据
2. 语法 : 必须声明的同时赋值
      ```javascript
      const PI = 3.14;
      ```
3. 注意 :
    + 常量一经定义,不能修改,强制修改会报错
    + 命名规范同变量,为了区分变量,常量名采用全大写字母

## 3. 数据类型
#### 1) 基本数据类型（简单数据类型）
1. number 数值类型
      + 整数
          1.   十进制表示
            ```javascript
             var a = 100;
            ```
            2. 八进制表示
               以0为前缀
            ```javascript
             var b = 021; //结果为十进制的 17
            ```
            3. 十六进制
               以0x为前缀
            ```javascript
             var c = 0x35;//结果为十进制的 53
            ```
               使用 : 整数可以采用不同进制表示,在控制台输出时一律会按照十进制输出
      + 小数
          1. 小数点表示
          ```javascript
           var m = 1.2345;
          ```
          2. 科学计数法
             例 : 1.5e3
              e表示10为底,e后面的数值表示10的次方数
             1.5e3 等价于 1.5 * 10(3)

2. string 字符串类型
   字符串 : 由一个或多个字符组成,使用""或''表示,每一位字符都有对应的Unicode编码
   ```javascript
   var s = "100";
   var s1 = "张三";
   ```


3. boolean 布尔类型
     只有真和假两个值，布尔值与number值可以互相转换。true 为 1，false 为 0

     ```javascript
     var isSave = true;
     var isChecked = false;
     ```

4. undefined  (程序返回的值)
     特殊值,变量声明未赋值时显示undefined
     ```javascript
     var a;
     console.log(a);//undefined
     ```

5. null 空类型  (主动使用的)
     解除对象引用时使用null,表示对象为空
#### 2) 引用数据类型
主要指对象，函数等
#### 3) 检测数据类型
typeof  变量或表达式
typeof (变量或表达式)


```javascript
var n = "asda";
console.log(typeof n);//string
console.log(typeof(n));//string
```
## 4. 数据类型转换
不同类型的数据参与运算时,需要转换类型
#### 1) 强制类型转换
1. 转换字符串类型
   方法 : toString()
   返回转换后的字符串
 ```javascript
 var a = 100;
 a = a.toString(); //"100"
 var b = true;
 b = b.toString(); //"true"
 ```
2. 转换number类型
    + Number(param)
        参数为要进行数据类型转换的变量或值，返回转换后的结果:
        	如果转换成功,返回number值
        	如果转换失败,返回NaN,(Not a Number)，只要数据中存在非number字符,一律转换失败，返回 NaN
    ```javascript
        Number("abc")
        typeof NaN
        Number(undefined)
        Number(null)
    ```

    + parseInt(param)
        参数为要解析的数据
            作用 : 从数据中解析整数值
            过程 :
                1. 如果参数为非字符串类型,会自动转成字符串
                        2. 从左向右依次对每一位字符转number,转换失败则停止向后解析,返回结果
    + parseFloat(param)
    	作用 : 提取number值，包含整数和小数部分

#### 2) 隐式类型转换（自动转换）
1. 当字符串与其他数据类型进行"+"运算时,表示字符串的拼接，不再是数学运算
   转换规则 ：将非字符串类型的数据转换成字符串之后进行拼接，最终结果为字符串

2. 其他情况下，一律将操作数转number进行数学运算
## 5. 运算符
#### 1) 赋值运算符 
	= 将右边的值赋给左边变量
#### 2) 算数运算符
	+ - * / %  加 减 乘 除 取余
#### 3) 复合运算符
	+= -= *= /= %=
#### 4) 自增或自减运算符
	++ -- 变量的自增和自减指的是在自身基础上进行 +1或-1 的操作
注意：
+ 自增或自减运算符在单独与变量结合时，放前和放后没有区别
+ 如果自增或自减运算符与其他运算符结合使用，要区分前缀和后缀,做前缀，那就先++/--,再进行赋值或其他运算，如果做后缀，就先结合其他运算符，再进行++ / --
#### 5) 关系运算符/比较运算符
```python
> <     
>= <=
==(相等) !=(不相等) # 值判断
===(全等) !==(不全等) # 值和类型一起判断
undefined == null # true
undefined === null # false
null > 1 # false  null在做大小判断的时候等于0
null == 0 # false  null在做等值判断的时候不等于0 
NaN和所有比较都是false
```
1. 关系运算符用来判断表达式之间的关系,结果永远是布尔值 true/false
2. 使用
	+ 字符串与字符串之间的比较
	依次比较每位字符的Unicode码,只要某位字符比较出结果,就返回最终结果
	+ 其他情况
	一律将操作数转换为number进行数值比较，如果某一操作数无法转换number，则变成NaN参与比较运算，结果永远是false

    null和其他数据类型做等值比较运算 不转换成数字
    null和undefined相等 但是 null和undefined不全等


3. 相等与全等
	+ 相等 : 不考虑数据类型,只做值的比较(包含自动类型转换)
	+ 全等 : 不会进行数据类型转换,要求数据类型一致并且值相等才判断全等
#### 6) 逻辑运算符 
1. && 逻辑与  条件1&&条件2  (and)
    表达式同时成立,最终结果才为true;全1则1
2. || 逻辑或  条件1||条件2   (or)
    表达式中只要有一个成立,最终结果即为true; 有1则1
3. ! 逻辑非    !条件  (not)
    对已有表达式的结果取反
    注意 : 除零值以外,所有值都为真
#### 7) 三目运算符
语法 :
```text
表达式1 ? 表达式2 : 表达式3;
```
过程 :
	判断表达式1是否成立,返回布尔值
	如果表达式1成立,执行表达式2;
	如果表达式1不成立,执行表达式3;

a>b?console.log('a比b大'):console.log('b比a大')
 1                				2                					3









# 一、流程控制
## 1. 作用
控制代码的执行顺序
## 2. 分类
#### 1）顺序结构
从上到下依次执行代码语句
#### 2）分支/选择结构
##### 1. if语句
+ 简单if结构
  ``` text
  if(条件表达式){
  	表达式成立时执行的代码段
  }
  ```
  注意 : 除零值以外，其他值都为真，以下条件为假值false
  ```javascript
  if(0){}
  if(0.0){}
  if(""){} //空字符串
  if(undefined){}
  if(NaN){}
  if(null){}
  ```
  特殊写法 :
  	{ }可以省略,一旦省略，if语句只控制其后的第一行代码
+ if - else结构
	```text
	if(条件表达式){
  	//条件成立时执行
  }else{
  	//条件不成立时选择执行
  }
  ```
+ 多重分支结构
    ```javascript
      if(条件1){
      	//条件1成立时执行
      }else if(条件2){
      	//条件2成立时执行
      }else if(条件3){
      	//条件3成立时执行
      }...else{
      	//条件不成立时执行
      }
    ```

![条件语句0](assets\条件语句0.png)
##### 2. switch语句

+ 语法 :
```javascript
switch(value){
	 case 值1 :
	 //value与值1匹配全等时,执行的代码段
	 break; //结束匹配
	 case 值2 :
	 //value与值2匹配全等时,执行的代码段
	 break;
	 case 值3 :
     //value与值3匹配全等时,执行的代码段
	 break;
	 default:
 	 //所有case匹配失败后默认执行的语句
 	 break;
}
```
+ 使用 :
```javascript
1. switch语句用于值的匹配，case用于列出所有可能的值；只有switch()表达式的值与case的值匹配全等时，才会执行case对应的代码段
2. break用于结束匹配，不再向后执行；可以省略，break一旦省略，会从当前匹配到的case开始，向后执行所有的代码语句，直至结束或碰到break跳出
3. default用来表示所有case都匹配失败的情况，一般写在末尾，做默认操作
4. 多个case共用代码段
  		case 值1:
  		case 值2:
  		case 值3:
  		//以上任意一个值匹配全等都会执行的代码段
```

![条件语句1](assets\条件语句1.png)
#### 3）循环结构

+ 作用
根据条件，重复执行某段代码
+ 分类
1. while循环
```text
定义循环变量;
   while(循环条件){
   条件满足时执行的代码段
   更新循环变量;
}
```
2. do-while循环
```text
do{
	循环体;
	更新循环变量
}while(循环条件);
```
![while语句3](assets\while语句3.png)

与 while 循环的区别 :

+ while 循环先判断循环条件，条件成立才执行循环体
+ do-while 循环不管条件是否成立，先执行一次循环体

3. for 循环
```text
for(定义循环变量;循环条件;更新循环变量){
	循环体;
}
```
![for循环语句2](assets\for循环语句2.png)

循环控制 :

1. break 强制结束循环
2. continue 结束当次循环，开始下一次循环
   循环嵌套 :
   在循环中嵌套添加其他循环

![循环中的语句4](assets\循环中的语句4.png)

# 二、函数

![函数定义0](assets/函数定义0.png)

## 1. 作用 
  封装一段待执行的代码

## 2. 语法 

```javascript
  //函数声明
  function 函数名(参数列表){
  	函数体
  	return 返回值;
  }
  //函数调用
  函数名(参数列表);
```

![函数定义带参数2](assets/函数定义带参数2.png)

## 3. 使用 

  	函数名自定义，见名知意，命名规范参照变量的命名规范。普通函数以小写字母开头，用于区分构造函数(构造函数使用大写字母开头，定义类)

![函数定义带返回值3](assets/函数定义带返回值3.png)

## 4. 匿名函数

匿名函数：省略函数名的函数。语法为：

- 匿名函数自执行

```javascript
 (function (形参){
  
 })(实参);
```

- 定义变量接收匿名函数

```javascript
 var fn = function (){};
 fn(); //函数调用
```

![匿名函数1](assets/匿名函数1.png)

## 5. 作用域

JavaScript 中作用域分为全局作用域和函数作用域，以函数的{ }作为划分作用域的依据

1. 全局变量和全局函数
   - 只要在函数外部使用 var 关键字定义的变量，或函数都是全局变量和全局函数，在任何地方都可以访问
   - 所有省略 var 关键字定义的变量，一律是全局变量
2. 局部变量/局部函数
   - 在函数内部使用 var 关键字定义的变量为局部变量，函数内部定义的函数也为局部函数，只能在当前作用域中使用，外界无法访问
3. 作用域链
   局部作用域中访问变量或函数，首先从当前作用域中查找，当前作用域中没有的话，向上级作用域中查找，直至全局作用域

![变量的作用域](assets/变量的作用域.png)

## 6. 获取多个DOM元素和控制属性

1. 根据标签名获取元素节点列表

```javascript
var elems = document.getElementsByTagName("");
/*
参数 : 标签名
返回值 : 节点列表,需要从节点列表中获取具体的元素节点对象,添加相应下标。
*/
```

2. 根据 class 属性值获取元素节点列表

```JavaScript
var elems = document.getElementsByClassName("");
/*
参数 : 类名(class属性值)
返回值 : 节点列表
*/
```

3. 元素节点对象提供了以下属性来操作元素内容

```text
innerHTML : 读取或设置元素文本内容,可识别标签语法
innerText : 设置元素文本内容,不能识别标签语法
value : 读取或设置表单控件的值
```

4. 获取 DOM 树中的属性值

![获取元素的属性值1](assets/获取元素的属性值1.png)

5. 设置 DOM 树中的属性值：

![设置元素的属性值0](assets/设置元素的属性值0.png)

```javascript
elem.getAttribute("attrname");//根据指定的属性名返回对应属性值  attrname一般是id或是class
elem.setAttribute("attrname","value");//为元素添加属性,参数为属性名和属性值
elem.removeAttribute("attrname");//移除指定属性
```









# 一、 内置对象
  ## 1） 对象
  对象是由属性和方法组成的,使用点语法访问
  ## 2） Array 数组

ArrayBuffer 数组扩展

  #### 1. 创建 
  #### 2. 特点 
+ #### JS 的数组像是python的字典，以“0”，“1”，“2”，……作为键

+ for (let i in 数组){}   取到的是字符串索引

+ 数组用于存储若干数据,自动为每位数据分配下标,从0开始

+ 数组中的元素不限数据类型,长度可以动态调整

+ 动态操作数组元素 ：根据元素下标读取或修改数组元素，arr[index]

+ JS的数组可以是不连续的，采取哈希映射存储，所以不能切片

+ 数组的索引值，本质上是字符串，JS自动把数字转成了字符串，所以使用数字和字符串都能定位元素
#### 3. 属性和方法
1. 属性 : length 表示数组长度,可读可写

    this.属性      谁调用方法，this就是谁

2. 方法 :
    + push(data)
    在数组的末尾添加一个或多个元素,多个元素之间使用逗号隔开
    返回添加之后的数组长度

    + pop()
    移除末尾元素
    返回被移除的元素

    + unshift(data)
    在数组的头部添加一个或多个元素
    返回添加之后的数组长度

    + shift()
    移除数组的第一个元素
    返回被移除的元素

    + splice(index,num,item)

      从数组中添加/删除项目

      返回被删除的项目

      item：（选填）在被删除的位置上添加内容
    
    + toString()
    将数组转换成字符串类型
      返回字符串结果
    
      保持原数组不变

    + join(param)
    
      转字符串：推荐使用方式

      将数组转换成字符串,可以指定元素之间的连接符,如果参数省略,默认按照逗号连接
      返回字符串
    
    + reverse()
      反转数组,倒序重排
      返回重排的数组,
    
      #### 注意该方法直接修改原数组的结构
    
    + sort(自定义排序规则，不写则按照2编码排序)
    
    内部规则：只接受返回值为数值，返回值>0则交换位置。
    
    	对数组中元素排序,默认按照Unicode编码升序排列
      返回重排后的数组,直接修改原有数组
      参数 : 可选,自定义排序算法
      例：
    
      ```javascript
        //自定义升序
        function sortASC(a,b){
          return a-b;
        }
      ```
       作用：作为参数传递到sort()中,会自动传入两个元素进行比较,如果a-b>0,交换元素的值,自定义升序排列
      ```javascript
        //自定义降序
        function sortDESC(a,b){
        	return b-a;
        }
        //如果返回值>0,交换元素的值,b-a表示降序排列
      ```
#### 4. 二维数组 
数组中的每个元素又是数组
```javascript
 var arr1 = [1,2,3];
 var arr2 = [[1,2],[3,4],[5,6,7]];
 //操作数组元素
 var r1 = arr2[0] //内层数组
 var num = r1[0]; //值 1
 //简写
 var num2 = arr2[1][0];
```
## 3）String 对象
#### 1. 创建 
```javascript
    var str = "100";
    var str2 = new String("hello");
```
#### 2. 特点 
字符串采用数组结构存储每位字符,自动为字符分配下标,从0开始
#### 3. 属性 
length ：获取字符串长度
#### 4. 方法 
+ 转换字母大小写
    toUpperCase() 转大写字母
    toLowerCase() 转小写字母
    返回转换后的字符串,不影响原始字符串

+ 获取字符或字符编码
    charAt(index)	   获取指定下标的字符
    charCodeAt(index)  获取指定下标的字符编码
    参数为指定的下标,可以省略,默认为0

+ 获取指定字符的下标

    + indexOf(str,fromIndex)
    作用 : 获取指定字符的下标,从前向后查询,找到即返回
    参数 :
    	str 表示要查找的字符串,必填
    	fromIndex 表示起始下标,默认为0
    返回 :
    	返回指定字符的下标,查找失败返回-1

+ 截取字符串
    substring(startIndex,endIndex)
    作用 : 根据指定的下标范围截取字符串,startIndex ~ endIndex-1
    参数 :
     startIndex	表示起始下标
     endIndex	表示结束下标,可以省略,省略表示截止末尾

+ substr(startIndex,len)

    作用：根据下标截取指定的字符串

+ 分割字符串
    split(param)
    作用 : 将字符串按照指定的字符进行分割,以数组形式返回分割结果
    参数 : 指定分隔符,必须是字符串中存在的字符,如果字符串中不存在,分割失败,仍然返回数组

+ 模式匹配
  
+ 正则表达式对象 RegExp
  
    RegExp : Regualr Expression
    
    1. 语法 ：
       var reg1 = /微软/ig;
       var reg2 = new RegExp('匹配模式','修饰符');
       正则表达式对象可以接收一个变量。
    
	2. 属性 ：
    
       lastIndex : 可读可写，表示下一次匹配的起始索引
       注意 ：
    
       1. 默认情况下，正则表达式对象不能重复调用方法，
          如果重复调用，结果会出错：
          由于 lastIndex 保存再一次匹配的起始下标，
          重复调用时，不能保证每次都从下标0开始
          验证，可以手动调整 lastIndex 为 0。
          
          ```js
          reg1.lastIndex = 0;
          ```
          
       2. 只有正则对象设置全局匹配 g ，该属性才起作用。
    
    3. 方法 ：
    
       test(str) :验证字符串中是否存在满足正则匹配模式的内容，存在则返回true，
    
       不存在返回false参数为要验证的字符串。
       
       ```js
       var reg1 = /this/ig;
       var str = "This is thiss mythis";
       console.log(reg1.test(str));//true
       ```
       
       
    
+ 作用 : 借助正则表达式实现字符串中固定格式内容的查找和替换
    正则表达式 :
     var reg1 = /字符模式/修饰符;
     修饰符 : 
      i :  ignorecase 忽略大小写
      g : global 全局范围
    字符串方法 :
    
    + match(regExp/subStr)
	作用 : 查找字符串中满足正则格式或满足指定字符串的内容
    	返回 : 数组,存放查找结果
    	
    	```js
    	var reg1 = /this/ig;
    	var str = "This is thiss mythis";
    	console.log(str.match("is"));//按字符串查找只能找到一个
    	//["is", index: 2, input: "This is thiss mythis", groups: undefined]
    	console.log(str.match(reg1));
    	//["This", "this", "this"]
    	```
    	
    	
    	
    + replace(regExp/subStr,newStr)
      作用 : 根据正则表达式或字符串查找相关内容并进行替换
      返回 : 替换后的字符串,不影响原始字符串。

## 4)  Math 对象

#### 1. 定义

Math对象主要提供一些列数学运算的方法

#### 2. 属性

1. 圆周率 :  Math.PI
2. 自然对数 : Math.E

#### 3. 方法

1. Math.random();   生成0-1之间的随机数
2. Math.ceil(x);	     对x向上取整,忽略小数位,整数位+1
3. Math.floor(x);      对x向下取整,舍弃小数位,保留整数位
4. Math.round(x);    对x四舍五入取整数

## 5）日期对象

#### 1. 创建日期对象

      1. var date2 = new Date("2011/11/11");
      2. var date3 = new Date("2011/11/11 11:11:11");

#### 2. 日期对象方法

1. 读取或设置当前时间的毫秒数：getTime()

2. 获取时间分量
   
   - getTime()         #获取毫秒数,可以直接用日期对象进行运算，等同于获取毫秒数的运算
   - getFullYear()    #获取年
   - getMonth()      #月份从0 开始计算
   - getDate()         #获取日
   - getDay()           #获取星期几
   - getHours()     #时
   - getMinutes()  #分
   - getSeconds()  #秒
   
3. 

   set…………    #设置时间





# 一、BOM 对象
## 1. BOM 介绍 
​		BOM全称为“Browser Object Model”，浏览器对象模型。提供一系列操作浏览器的属性和方法。核心对象为window对象，不需要手动创建，跟随网页运行自动产生，直接使用，在使用时可以省略书写。
## 2. 对象方法
1. ##### 网页弹框

   ```javascript
   alert()		//警告框
   confirm()	//确认框 返回值是true和false
   ```

2. ##### 定时器方法

![setInterval函数2](assets\setInterval函数2.png)

**周期性定时器**
	作用：每隔一段时间就执行一次代码

```javascript
//开启定时器:
var timerID = setInterval(function,interval);
/*
参数 :
 function : 需要执行的代码,可以传入函数名;或匿名函数
 interval : 时间间隔,默认以毫秒为单位 1s = 1000ms
返回值 : 返回定时器的ID,用于关闭定时器
*/
```
   关闭定时器 :
```javascript
//关闭指定id对应的定时器
clearInterval(timerID);
```
![setTimeout函数3](assets\setTimeout函数3.png)

**一次性定时器**
	作用：等待多久之后执行一次代码

```javascript
//开启超时调用:
var timerId = setTimeout(function,timeout);
//关闭超时调用:
clearTimeout(timerId);
```
## 3. 对象属性
window的大部分属性又是对象类型

1. ##### history

   作用：保存当前窗口所访问过的URL
   属性 :  length 表示当前窗口访问过的URL数量
   方法 :

   ```javascript
   back() 对应浏览器窗口的后退按钮，访问前一个记录
   forward() 对应前进按钮，访问记录中的下一个URL
   // 手机界面用的比较多
   ```

2. ##### location

   作用：保存当前窗口的地址栏信息(URL) 
   属性 : 
   
   ```js
   location.href //从JS跳转页面
   ```
   
   

​			前端后台跳转页面，无需超链接被点击

   方法 :

   ```js
   reload(param) //重载页面(刷新)
   //参数为布尔值，默认为 false，表示从缓存中加载，设置为true,强制从服务器根目录加载
   // 尽量少用，而且尽量不要放在定时函数里，每一次刷新都是一次请求和回复，对服务器有较大的压力
   ```

# 二、DOM节点操作
DOM全称为 “Document Object Model”，文档对象模型，提供操作HTML文档的方法。（注：每个html文件在浏览器中都视为一篇文档,操作文档实际就是操作页面元素。）
## 1. 节点对象

JavaScript 会对 html 文档中的元素、属性、文本甚至注释进行封装，称为节点对象，提供相关的属性和方法。

## 2. 访问节点

- 元素节点   ( 操作标签）
- 属性节点（操作标签属性）
- 文本节点（操作标签的文本内容）

标签属性都是元素节点对象的属性,可以使用点语法访问，例如：

```javascript
h1.id = "d1"; 		 //set 方法
console.log(h1.id);  //get 方法
h1.id = null;		//remove 方法
```

```js
Element.onmouseover = function(){} //鼠标移入
Element.onmouseout = function(){} //鼠标移出
```

注意 :

- 属性值以字符串表示
- class属性需要更名为 className，避免与关键字冲突，例如：
  h1.className = "c1 c2 c3"；

## 3. 操作元素样式

1. 为元素添加 id、class属性，对应选择器样式

   div.className = ""

2. 操作元素的行内样式，访问元素节点的style属性，获取样式对象；样式对象中包含CSS属性，使用点语法操作。

```javascript
p.style.color = "white";
p.style.width = "300px";
p.style.fontSize = "20px";
```

注意 :

- 属性值以字符串形式给出，单位不能省略
- 如果css属性名包含连接符，使用JS访问时，一律去掉连接符,改为驼峰， font-size -> fontSize



# jQuery简介
## 1. 介绍 
jQuery是JavaScript的工具库，对原生JavaScript中的DOM操作、事件处理、包括数据处理和Ajax技术等进行封装,提供更完善，更便捷的方法。

兼容性更好

1.几版本比较适用PC

2-3.几版本适用于手机和平板

## 2. 使用 
#### 1）引入
先引入jquery文件，才能使用jquery语法

1. CDN 有网（备用）
2. 本地文件（常用）

#### 2）工厂函数 - $()
"$()"函数用于获取元素节点，创建元素节点或将原生JavaScript对象转换为jquery对象,返回 jQuery 对象。jQuery 对象实际是一个类数组对象，包含了一系列 jQuery 操作的方法。
例如 :

```javascript
 //$()获取元素节点,需传入字符串的选择器
 $("h1") 
 $("#d1") 
 $(".c1") 
 $("body,h1,p") 
 //选择器的特点，与样式选择器一致
```
#### 3）原生JavaScript对象与jQuery对象
原生JavaScript对象与jQuery对象的属性和方法不能混用。可以根据需要，互相转换 :
1. 原生JavaScript转换jQuery对象
    $(原生对象)，返回 jQuery 对象
2. jQuery对象转换原生JavaScript对象
    + 方法一 : 根据下标取元素,取出即为原生对象
      var div = $("div")[0];
    + 方法二 : 使用jQuery的get(index)取原生对象
      var div2 = $("div").get(0);
#### 4）jQuery获取元素
jQuery通过选择器获取元素，$("选择器")
选择器分类 :

1. 基础选择器
```text
标签选择器：$("div")
ID 选择器：$("#d1")
类选择器：$(".c1")
群组选择器：$("body,p,h1")
```
2. 层级选择器
```text
后代选择器： $("div .c1")
子代选择器： $("div>span")
相邻兄弟选择器： $("h1+p")  匹配选择器1后的第一个兄弟元素,同时要求兄弟元素满足选择器2
通用兄弟选择器： $("h1~h2") 匹配选择器1后所有满足选择器2的兄弟元素
$(this).siblings()   //找到当前元素得所有兄弟元素
```
3. 过滤选择器，需要结合其他选择器使用。
```text
:first
  匹配第一个元素 例:$("p:first")
:last
  匹配最后一个元素 例:$("p:last")
:odd
  匹配奇数下标对应的元素
:even
  匹配偶数下标对应的元素
:eq(index)
  匹配指定下标的元素
:lt(index)
  匹配下标小于index的元素
:gt(index)
  匹配下标大于index的元素
:not(选择器)
  否定筛选,除()中选择器外,其他元素
```
#### 5）操作元素内容
```javascript
html() //设置或读取标签内容,等价于原生innerHTML,可识别标签语法
text() //设置或读取标签内容,等价于innerText,不能识别标签
val()  //设置或读取表单元素的值,等价于原生value属性
```
#### 6）操作标签属性
1. attr("attrName","value")
    设置或读取标签属性
    
2. prop("attrName","value")
    设置或读取标签属性
    注意 :在设置或读取元素属性时,attr()和prop()基本没有区别;但是在读取或设置表单元素(按钮)的选中状态时,必须用prop()方法,attr()不会监听按钮选中状态的改变,只看标签属性checked是否书写
    
    监听checked属性返回值是布尔值
    
3. removeAttr("attrName")
    移除指定属性
#### 7）操作标签样式
1. 为元素添加id/class属性,对应选择器样式
2. 针对类选择器,提供操作class属性值的方法
```javascript
addClass("className")	//添加指定的类名
removeClass("className")//移除指定的类型,如果参数省略,表示清空class属性值
toggleClass("className")//结合用户行为,实现动态切换类名.如果当前元素存在指定类名,则移除;不存在则添加
//在jquery中使用this.，需要写成：   $(this).
```
3. 操作行内样式
```javascript
css("属性名","属性值")  //设置行内样式
css(JavaScriptON对象)			 //设置一组CSS样式，不建议使用，如果有一大段代码要改，建议使用类
/*
JavaScriptON对象:常用数据传输格式
语法 :
   {
    "width":"200px",
    "height":"200px",
    "color":"red"
   }
 */
```
#### 8）元素的创建,添加,删除
1. 创建：使用$("标签语法")，返回创建好的元素
```javascript
var div = $("<div></div>");	//创建元素
div.html("动态创建").attr("id","d1").css("color","red"); //链式调用，设置内容和属性
var h1 = $("<h1 id='d1'>一级标题</h1>");	//创建的同时设置内容，属性和样式
```
2. 作为子元素添加
```javascript
$obj.append(newObj);	//在$obj的末尾添加子元素newObj
$obj.prepend(newObj);	//作为第一个子元素添加至$obj中
```
3. 作为兄弟元素添加

```javascript
$obj.after(newObj);		//在$obj的后面添加兄弟元素
$obj.before(newObj);	//在$obj的前面添加兄弟元素
```
4. 移除元素 

```javascript
$obj.remove();	//移除$obj
```
#### 9）动画效果

1. 显示和隐藏

   ```javascript
    show(speed,callback)/hide(speed,callback)
   ```

   - speed	  可选。规定元素从隐藏到完全可见的速度。默认为 "0"。
   - callback   可选。show 函数执行完之后，要执行的函数

2. 通过使用滑动下拉和上推效果，显示隐藏的被选元素（ **只针对块元素** ）

   ```javascript
    slideDown(speed,callback)/slideUp(speed,callback)
   ```

3. 通过使用淡隐淡现方式显示效果，显示隐藏的被选元素

   ```javascript
    fadeOut(speed,callback)/fadeIn(speed,callback)
   ```

4. 动画函数，可以实现自定义动画 animate 函数

   ```javascript
   animate(styles,speed,callback)
   ```

   - styles    必需。规定产生动画效果的 CSS 样式和值
   - speed   可选。规定动画的速度。默认是 "normal"
   - callback   可选。show 函数执行完之后，要执行的函数

#### 10）数据与对象遍历

1. $(selector).each() 方法规定为每个匹配元素规定运行的函数

   ```javascript
   $(selector).each(function(index,element){})
   ```

   必需。为每个匹配元素规定运行的函数。

   - *index* - 选择器的 index 位置
   - element - 当前的元素，这个当前元素有可能是DOM对象

2. $.each()函数是框架提供的一个工具类函数，通过它，你可以遍历对象、数组的属性值并进行处理

   ```javascript
   $.each(Object, function(index, data){});
   ```

   必需。为每个匹配元素规定运行的函数。

   - *index* - 选择器的 index 位置
   - data- 当前的数据
#### 11）jQuery事件处理

1. 文档加载完毕
   原生JavaScript 方法：window.onload
   jQuery:

```javascript
//语法一 
$(document).ready(function (){
  //文档加载完毕后执行
})
//语法二 
$().ready(function (){
  //文档加载完毕后执行
})
//语法三 
$(function(){
  //文档加载完毕后执行
})
```

区别：
原生onload事件不能重复书写，会产生覆盖问题；jquery中对事件做了优化,可以重复书写ready方法,依次执行

2. 事件绑定方式
   事件名称省略 on 前缀

```javascript
  //on("事件名称"，function)
  $("div").on("click",function(){});//新版本使用的多些
  //bind("事件名称",function)
  $("div").bind("click",function(){});//1.6-1.8间的版本
  //事件名作为方法名
  $("div").click(function(){});  
```

3. this表示事件的触发对象，在jquery中可以使用，注意转换类型。this为原生对象只能使用原生的属性和方法，可以使用$(this)转换为jquery对象，使用jquery方法。

## 3.实战

### 	1. 页面效果

![作业1](assets\作业1.png)

###     2. 代码分析

####             1. 页面元素

​                   ![1565678581771](assets\1565678581771.png)

#### 		    2. 初始代码

​                   ![1565682366493](assets\1565682366493.png)

####             3. 绑定省份

​                    ![1565682409353](assets\1565682409353.png)

####              4. 绑定城市

​                    ![1565682448560](assets\1565682448560.png)