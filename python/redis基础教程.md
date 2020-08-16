## **60Redis-day01-note**

**Redis介绍**

- **特点及优点**

```python
1、开源的，使用C编写，基于内存且支持持久化
2、高性能的Key-Value的NoSQL数据库
3、支持数据类型丰富，字符串strings，散列hashes，列表lists，集合sets，有序集合sorted sets 等等
4、支持多种编程语言（C C++ Python Java PHP ... ）
5、单进程单线程
```

- **与其他数据库对比**

```python
1、MySQL : 关系型数据库，表格，基于磁盘，慢
2、MongoDB：键值对文档型数据库，值为类似JSON文档，数据结构相对单一
3、Redis的诞生是为了解决什么问题？？
   # 解决硬盘IO带来的性能瓶颈
```

- **应用场景**

```python
1，缓存
2，并发计数
3，排行榜
4，生产者消费者模型
...
```

- **redis版本**

```python
1、最新版本：5.0
2、常用版本：2.4、2.6、2.8、3.0(里程碑)、3.2、3.4、4.0(教学环境版本)、5.0
```

- **Redis附加功能**

```python
1、持久化
  将内存中数据保存到磁盘中，保证数据安全，方便进行数据备份和恢复
2、过期键功能
   为键设置一个过期时间，让它在指定时间内自动删除
   <节省内存空间>
   # 音乐播放器，日播放排名，过期自动删除
3、事务功能
   原子的执行多个操作
4、主从复制
5、Sentinel哨兵
```

## **安装**

- Ubuntu

```python
# 安装
sudo apt-get install redis-server
# 服务端启动
sudo /etc/init.d/redis-server status | start | stop | restart
# 客户端连接
redis-cli -h IP地址 -p 6379 -a 密码
```

## **配置文件详解**

- **配置文件所在路径**

```python
/etc/redis/redis.conf
mysql的配置文件在哪里？ : /etc/mysql/mysql.conf.d/mysqld.cnf
```

- **设置连接密码**

```python
1、requirepass 密码
# 服务器redis必须有密码，否则会被黑客攻击
2、重启服务
   sudo /etc/init.d/redis-server restart
3、客户端连接
   redis-cli -h 127.0.0.1 -p 6379 -a 123456
   127.0.0.1:6379>ping
```

- **允许远程连接**

```python
1、注释掉本地IP地址绑定
	set number 查看行号
  69行: # bind 127.0.0.1 ::1
2、关闭保护模式(把yes改为no)
  88行: protected-mode no
3、重启服务
  sudo /etc/init.d/redis-server restart
```

- **通用命令 ==适用于所有数据类型==**

```python
# 切换库(number的值在0-15之间,db0 ~ db15)
select number
# 查看键
keys 表达式  # keys *
# 数据类型
type key
# 键是否存在
exists key
# 删除键
del key
# 键重命名
rename key newkey
# 清除当前库中所有数据（慎用）
flushdb
# 清除所有库中所有数据（慎用）
flushall
```

django连接redis

```python

# from django.core.cache import cache
	# cache.set('key','value',time)   这个value可以传入任何python的对象，django会对其进行转化，将其存入
# from django_redis import get_redis_connection
	# r = get_redis_connection()
	# r.                              这个只能存字符串
settings里面添加
CACHES = {
    "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": "redis://127.0.0.1:6379/1",    # /1是redis库
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
                "PASSWORD":"mu7401889",		# 如果没有密码，请注释掉这一行
            }
        }
}



# 建立连接
conn = get_redis_connection('default')
# 存
conn.rpush(key_name, str({"api_key":"1234"})) # 将 {"api_key":"1234"}这个字典作为value 存入键为key_name的列表中
#即：key_name：[{"api_key":"1234"},{"two_key":"4567"},...]
# 取
data = conn.blpop(key_name) # data是一个二进制元组，（b'key_name, b'{"api_key":"1234"}）
#所以如果想得到value值，得先进行解包
key_name, value_dict= data
value_dict = b'{"api_key":"1234"}
#需要强调一下，任何从redis取出来的数据，都是二进制，要先进行二进制解码
value_dict  = value_dict.decode("UTF-8")
#解码后value_dict的类型是string, 如果想要得到其中字典的值，就需要进行转换
value_dict = eval(value_dict) # eval可以智能地根据字符串中的数据类型进行转换。


```



## 数据类型

### **字符串类型(string)**

- **特点**

```python
1、字符串、数字，都会转为字符串来存储
2、以二进制的方式存储在内存中
```

- **字符串常用命令-==必须掌握==**

```python
# 1. 设置一个key-value
set key value
# 2. 获取key的值
get key
# 3. key不存在时再进行设置(nx)
set key value nx  # not exists
# 4. 设置过期时间(ex)
set key value ex seconds

# 5. 同时设置多个key-value
mset key1 value1 key2 value2 key3 value3
# 6. 同时获取多个key-value
mget key1 key2 key3 
```

- **字符串常用命令-==作为了解==**

```python
# 1.获取长度
strlen key
# 2.获取指定范围切片内容 [包含start stop]
getrange key start stop
# 3.从索引值开始，value替换原内容
setrange key index value
```

- **数值操作-==字符串类型数字(必须掌握)==**

```python
# 整数操作
incrby key 步长	# 没有这个KEY则创建这个key,值就是步长
decrby key 步长
incr key : +1操作
decr key : -1操作
# 应用场景: 抖音上有人关注你了，是不是可以用INCR呢，如果取消关注了是不是可以用DECR
# 浮点数操作: 自动先转为数字类型，然后再进行相加减，不能使用append
incrbyfloat key step
```

- **string命令汇总**

```python
# 字符串操作
1、set key value
2、set key value nx
3、get key
3、mset key1 value1 key2 value2
4、mget key1 key2 key3
5、set key value nx ex seconds
6、strlen key 
# 返回旧值并设置新值（如果键不存在，就创建并赋值）
7、getset key value
# 数字操作
7、incrby key 步长
8、decrby key 步长
9、incr key    # +1  r.incr('bbbb')  键不存在时创建一个键，并置1
10、decr key
11、incrbyfloat key number#(可为正数或负数)

# 设置过期时间的两种方式
# 方式一
1、set key value ex 3   # 推荐使用方式
# 方式二
1、set key value
2、expire key 5 # 秒
3、pexpire key 5 # 毫秒
# 查看存活时间
ttl key   # 返回 -1：当前KEY存在，没有过期时间   -2：当前KEY不存在   >0：当前剩余时间
# 删除过期
persist key
```

- **string数据类型注意**

```python
# key命名规范
可采用 - wang:email
# key命名原则
1、key的名字不宜过长，消耗内存，且在数据中查找这类键值的计算成本高
2、不宜过短，可读性较差
# 值
1、一个字符串类型的值最多能存储512M内容
```

- 业务场景
  - 缓存
    - 将mysql中的数据存储到redis字符串类型中
  - 并发计数 - 点赞/秒杀
    - 说明：通过redis单进程单线程的特点，由redis负责计数，并发问题转为串行问题
  - 带有效期的验证码  - 短信验证码
    - 借助过期时间，存放验证码；到期后，自动消亡

**练习**

```python
1、查看 db0 库中所有的键
2、设置键 trill:username 对应的值为 user001，并查看
3、获取 trill:username 值的长度
4、一次性设置 trill:password 、trill:gender、trill:fansnumber 并查看（值自定义）                 
5、查看键 trill:score 是否存在
6、增加10个粉丝
7、增加2个粉丝（一个一个加）
8、有3个粉丝取消关注你了
9、又有1个粉丝取消关注你了
10、思考、思考、思考...,清除当前库
11、一万个思考之后，清除所有库
```

### **列表数据类型（List）**

- **特点**

```python
1、元素是字符串类型 #【只能是一维列表】
2、列表头尾增删快，中间增删慢，增删元素是常态   #【双向操作，不要在中间操作】
3、元素可重复
4、最多可包含2^32 -1个元素
5、索引同python列表
```

- **列表常用命令**

```python
# 增
1、从列表头部压入元素
	LPUSH key value1 value2 
    返回：list长度
2、从列表尾部压入元素
	RPUSH key value1 value2
    返回：list长度
    
 	rpushx 在指定的列表尾部插入一个元素，若该列表不存在则操作无效
		127.0.0.1:6379> rpush list4 hello
        (integer) 1
        127.0.0.1:6379> rpushx list5 hello
        (integer) 0
3、从列表src尾部弹出1个元素,压入到列表dst的头部
	RPOPLPUSH src dst
    返回：被弹出的元素
    【数据在两个列表中移动，必须使用这个命令，能保证这两步能做到位，一旦分开执行，就有失败的风险】
4、在列表指定元素后/前插入元素【不推荐使用，慢】
	LINSERT key after|before value newvalue
    返回：
		1，如果命令执行成功，返回列表的长度
		2，如果没有找到 pivot ，返回 -1 【一旦返回 -1 ，说明列表被全部遍历了一遍，极其消耗时间】
		3，如果 key 不存在或为空列表，返回 0 

# 查
5、查看列表中元素
	LRANGE key start stop
  # 查看列表中所有元素: LRANGE key 0 -1
r.lrange('aaaa', 1, 1)  # 当请求的列表不存在或者索引不存在时，返回空列表[]

	lindex
	# 返回list中指定下标的元素,0是开始下标,如果下标位负数,那么顺序从后往前,-1是开始位置也是列表中的最后元素
lindex list2 1

#在start位负数时,它的起点位置从后面开始数,所以说,start确定了值的时候,lrange可以输出的最大范围已经决定了,不管stop的值是什么,输出的元素范围绝对是start开始往后的元素最大范围.
#所以Redis中的range相关命令其实最重要要看start参数的值,stop只是决定了尾部在哪里

#在lrange中,stop的值绝对不能在start的前面
6、获取列表长度
	LLEN key

# 删
7、从列表头部弹出1个元素
	LPOP key
8、从列表尾部弹出1个元素
	RPOP key
9、列表头部,阻塞弹出,列表为空时阻塞
	BLPOP key timeout
10、列表尾部,阻塞弹出,列表为空时阻塞
	BRPOP key timeout
  # 关于BLPOP 和 BRPOP
  	1、如果弹出的列表不存在或者为空，就会阻塞
	2、超时时间设置为0，就是永久阻塞，直到有数据可以弹出
	3、如果多个客户端阻塞再同一个列表上，使用First In First Service原则，先到先服务
    4、如果超时，则GG
    	127.0.0.1:6379> blpop list2 10
        (nil)
        (10.09s) #超时
11、删除指定元素【不推荐使用】
	LREM key count value
    count>0：表示从头部开始向表尾搜索，移除与value相等的元素，数量为count
	count<0：表示从尾部开始向表头搜索，移除与value相等的元素，数量为count
	count=0：移除表中所有与value相等的值
    返回：被移除元素的数量
    
12、保留指定范围内的元素
	LTRIM key start stop
    返回：ok
    样例：
  		LTRIM mylist1 0 2 # 只保留前3条
  		# 应用场景: 保存微博评论最后500条
  		LTRIM weibo:comments 0 499
# 改
13、将列表 key 下标为 index 的元素的值设置为 value
	LSET key index newvalue
```

**练习**

```python
1、查看所有的键
2、向列表 spider:urls 中以RPUSH放入如下几个元素：01_baidu.com、02_taobao.com、03_sina.com、04_jd.com、05_xxx.com
3、查看列表中所有元素
4、查看列表长度
5、将列表中01_baidu.com 改为 01_tmall.com
6、在列表中04_jd.com之后再加1个元素 02_taobao.com
7、弹出列表中的最后一个元素
8、删除列表中所有的 02_taobao.com
9、剔除列表中的其他元素，只剩前3条
```

## **python交互redis**

## redis删除过期KEY的策略

```
redis删除过期KEY的策略
1,主动出击
	1,将带过期时间的key存到-一个独立的字典中
		默认每100毫秒进行一次过期扫描
		1.在过期字典中随机20个KEY
		2.检查过期时间，删除已经过期的KEY
		3.如果过期的KEY的比例超过 1/4 重复1-3
		1-3步骤最多占用25毫秒
	问题：大量KEY同时过期，造成redis明显卡顿
	解决方案：在创建数据时，设置过期时间时，对该时间做小量的偏移
			set key value ex 1000 + random.ranint(1,100)
2,惰性删除
	1, get 【检查key的过期时间，过期则删除】
	2，set  maxmemory【最大内存检查】
		noeviction	拒绝写服务，可以接受读请求【默认配置】
		volatile-lru 尝试淘汰设置了过期时间的KEY，淘汰最近很少使用的KEY（没过期的也可能删除）【最少使用原则】（手写版lru算法（堆栈））（redis怎么知道哪些KEY最少使用）
		allkeys-lru	 尝试淘汰所有KEY

```



- **模块(redis)**

Ubuntu

```python
sudo pip3 install redis
```

- **使用流程**

```python
import redis
# 创建数据库连接对象
r = redis.Redis(host='127.0.0.1',port=6379,db=0,password='123456')
```

- **通用命令代码示例**

```python

```

- **python操作list**

```python

```

**list案例: 一个进程负责生产任务，一个进程负责消费任务**

中间那个redis-list：降低了服务器之间的沟通成本【消息队列】

进程1: 生产者

```python

```

进程2: 消费者

```python
# 判断 有没有django的环境
if not os.getenv('DJANGO_SETTINGS_MODULE'):
	os.environ['DJANGO_SETTINGS_MODULE']='settings'

```

- **python操作string**

```python
r.mset({'key':'value','key2':'value2',……})  # 批量插入
r.mget('key','key2')
r.incry('key')     # 这两个方法在pyredis里面都指向同一个方法，建议按redis终端的方式进行调用
r.incryby('key',10)
```









# **redis_day02笔记**

## **位图操作bitmap**

**定义**

```python
1、位图不是真正的数据类型，它是定义在字符串类型中
2、一个字符串类型的值最多能存储512M字节的内容，位上限：2^32
# 1MB = 1024KB
# 1KB = 1024Byte(字节)
# 1Byte = 8bit(位)
```

**强势点**

```python
可以实时的进行统计，极其节省空间。官方在模拟1亿2千8百万用户的模拟环境下，在一台MacBookPro上，典型的统计如“日用户数”的时间消耗小于50ms, 占用16MB内存
```

**SETBIT 命令**

- 说明：设置某位置上的二进制值

- 语法：SETBIT key offset value

- 参数：offset - 偏移量 从0开始

  			value - 0或者1

- 示例：

```python
# 默认扩展位以0填充
127.0.0.1:6379> SET mykey ab
OK
127.0.0.1:6379> GET mykey
"ab"
127.0.0.1:6379> SETBIT mykey 0 1
(integer) 0
127.0.0.1:6379> GET mykey
"\xe1b"
127.0.0.1:6379> 
```

**GETBIT 命令**

- 说明：获取某一位上的值

- 语法：GETBIT key offset

- 示例：

```python
127.0.0.1:6379> GETBIT mykey 3
(integer) 0
127.0.0.1:6379> GETBIT mykey 0
(integer) 1
127.0.0.1:6379> 
```

**BITCOUNT 命令**

- 说明：统计键所对应的值中有多少个 1 
- 语法：BITCOUNT key start end
- 参数：start/end 代表的是 **字节索引**
- 示例：

```python
127.0.0.1:6379> SET mykey1 ab
OK
127.0.0.1:6379[4]> BITCOUNT mykey
(integer) 6
127.0.0.1:6379[4]> BITCOUNT mykey 0 0
(integer) 3

```

**应用场景案例**

```python
# 网站用户的上线次数统计（寻找活跃用户）
	用户名为key，上线的天作为offset，上线设置为1
# 示例
	用户名为 user1:login 的用户，今年第1天上线，第30天上线
	SETBIT user1:login 0 1 
	SETBIT user1:login 29 1
	BITCOUNT user1:login
```

## RedisBloom 布隆过滤

**多用于解决缓存穿透**

**Redis中的布隆过滤器实现（rebloom模块扩展）**

下载并编译
git clone git://github.com/RedisLabsModules/rebloom
cd rebloom
make
配置文件中加载rebloom
loadmodule /your_path/rebloom.so
重启Redis服务器即可
./bin/redis-cli -h 127.0.0.1 -p 6379 -a ****** shutdown
./bin/redis-server redis.conf

 

**rebloom在Redis中的使用**

**bloom filter定义**

BF.RESERVE {key} {error_rate} {capacity}
使用给定的期望错误率和初始容量创建空的Bloom过滤器（如果不存在的话）。如果打算向Bloom过滤器中添加许多项，则此命令非常有用，否则只能使用BF.ADD 添加项。
初始容量和错误率将决定过滤器的性能和内存使用情况。一般来说，错误率越小(即对误差的容忍度越低)，每个过滤器条目的空间消耗就越大。

**bloom filter基本操作**

1，BF.ADD {key} {item}
单条添加元素
向Bloom filter添加一个元素，如果该key不存在，则创建该key(过滤器)。
如果项是新插入的，则为“1”;如果项以前可能存在，则为“0”。

2，BF.MADD {key} {item} [item...]
批量添加元素
布尔数(整数)的数组。返回值为0或1的范围的数据，这取决于是否将相应的输入元素新添加到过滤器中，或者是否已经存在。

3，BF.EXISTS {key} {item}
判断单个元素是否存在
如果存在，返回1，否则返回0

4，BF.MEXISTS {key} {item} [item...]
判断多个元素是否存在
布尔数(整数)的数组。返回值为0或1的范围的数据，这取决于是否将相应的元是否已经存在于key中。

5，bf.insert

bf.insert{key} [CAPACITY {cap}] [ERROR {ERROR}] [NOCREATE] ITEMS {item…}
该命令将向bloom过滤器添加一个或多个项，如果它还不存在，则默认情况下创建它。有几个参数可用于修改此行为。
key:过滤器的名称
capacity:如果指定了，应该在后面加上要创建的过滤器的所需容量。如果过滤器已经存在，则忽略此参数。如果自动创建了过滤器，并且没有此参数，则使用默认容量(在模块级指定)。见bf.reserve。
error:如果指定了，后面应该跟随着新创建的过滤器的错误率(如果它还不存在)。如果自动创建过滤器而没有指定错误，则使用默认的模块级错误率。见bf.reserve。
nocreate:如果指定，表示如果过滤器不存在，就不应该创建它。如果过滤器还不存在，则返回一个错误，而不是自动创建它。如果需要在创建过滤器和添加过滤器之间进行严格的分离，可以使用这种方法。将NOCREATE与容量或错误一起指定是一个错误。
item:指示要添加到筛选器的项的开头。必须指定此参数。

**bf持久化操作**

BF.SCANDUMP {key} {iter}

对bloom过滤器进行增量保存。这对于不能适应常规save和restore模型的大型bloom filter非常有用。
第一次调用这个命令时，iter的值应该是0。这个命令将返回连续的(iter, data)对，直到(0,NULL)，以表示完成
python伪代码演示:

```python
# python伪代码演示:
chunks = []
iter = 0
while True:
    iter, data = BF.SCANDUMP(key, iter)
    if iter == 0:
        break
    else:
        chunks.append([iter, data])

# Load it back
for chunk in chunks:
    iter, data = chunk
    BF.LOADCHUNK(key, iter, data)

```

```redis
# bf.scandump示例

127.0.0.1:8001> bf.scandump bloom_filter_test2 0
1) (integer) 1
2)"\x06\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00{\x14\xaeG\xe1z\x84?\x88\x16\x8a\xc5\x8c+#@\a\x00\x00\x00j\x00\x00\x00\n"
127.0.0.1:8001> bf.scandump bloom_filter_test2 1
1) (integer) 129
2) "\x00\x00\x00\x00\xa2\x00\x00\x00\x00\x00\x00B\x01\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00 \x00\x00\b\x00\x00\x00\x00\b\x00\x00@\x00\x01\x04\x18\x02\x00\x00\x00\x82\x00\x00\x80@\x00\b\x00\x00\x00\x00 \x00\x00@\x00\x00\x00\x00\x18\b\x00\b\x00\b\x00\x80B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00 (\x00\x00\x00\x00@\x00\x00\x00\x00@\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x80\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\b"
127.0.0.1:8001> bf.scandump bloom_filter_test2 129
1) (integer) 0
2) ""
127.0.0.1:8001>
```









## **Hash散列数据类型**

- **定义**

```python
1、由field和关联的value组成的键值对
2、field和value是字符串类型
3、一个hash中最多包含2^32-1个键值对
```

- **优点**

```python
1、节约内存空间 - 特定条件下 【1，字段小于512个（超过512再压缩就比较费CPU，不划算了），2：value不能超过64字节】
2、可按需获取字段的值
```

- **缺点（不适合hash情况）**

```python
1，使用过期键功能：键过期功能只能对键进行过期操作，而不能对散列的字段进行过期操作
2，存储消耗大于字符串结构
```

- **基本命令操作**

```python
# 1、设置单个字段
HSET key field value
HSETNX key field value
# 2、设置多个字段
HMSET key field value field value
# 3、返回字段个数
HLEN key
# 4、判断字段是否存在（不存在返回0）
HEXISTS key field
# 5、返回字段值
HGET key field
# 6、返回多个字段值
HMGET key field filed
# 7、返回所有的键值对
HGETALL key
# 8、返回所有字段名
HKEYS key
# 9、返回所有值
HVALS key
# 10、删除指定字段
HDEL key field 
# 11、在字段对应值上进行整数增量运算
HINCRBY key filed increment  # 返回值是增量后的结果   如果没有这个字段则创建这个字段
# 12、在字段对应值上进行浮点数增量运算
HINCRBYFLOAT key field increment
```

**python操作hash**

MYSQL 更新数据时，对应的缓存整条删除，有访问此条数据时再添加回去

```python
对属性设置的时候用字符串，获取的时候用字节串
name 始终都是字符串类型
# 1、更新一条数据的属性，没有则新建
hset(name, key, value)
# 2、读取这条数据的指定属性， 返回字符串类型
hget(name, key)
# 3、批量更新数据（没有则新建）属性,参数为字典
hmset(name, dict)   # 这个 dict 里面是字符串
# 4、批量读取数据（没有则新建）属性
hmget(name, keys)   # 返回的字典里面键值对都是字节串 可以用字典推导式转成字符串的字典
# 5、获取这条数据的所有属性和对应的值，返回字典类型
hgetall(name)	#{b'user': b'word', b'user2': b'word2'}
# 6、获取这条数据的所有属性名，返回列表类型	
hkeys(name)
# 7、删除这条数据的指定属性
hdel(name, *keys)
# 8. 删除整条
delete(name)
# 9.加过期时间
expire(name,time)
```

```
redis hash的实现：
	python：
		a = {}   # 初始化了一个长度为8的数组
		a['a1'] = 123	# 对KEY进行哈希运算： hash('a1') % 8 = index,     然后创建一个包含hash值，KEY，VUALUE的对象放在index对应的索引上。 
		a['a2'] = 456	# 如果发生哈希碰撞：	开放地址法（python）：以（当前位置+偏移量）进行二次哈希运算，直到不碰撞为止（只要有空位，就一定能找到，由算法决定的）。
		扩容：超过2/3（python），1）总量小于50000-->扩4倍。2）总量大于50000-->扩2倍，-------->扩容之后，所有元素重新计算哈希值（rehash），重排
		查找：由于使用开放地址法，找到空座位就会返回KEYerror，如果某个位置上的元素被删除了，那么会在原来的位置上留下标记（告诉python这里曾经有人，让python继续往下找。）
	redis:
		如果发生哈希碰撞:（单链法）在该位置上生成一个新的链表直接存入
		扩容：		元素总量/len() > 1 有可能扩容， > 5 肯定扩容，扩大到：2**n > 元素总量*2
		rehash:		渐进式rehash  
        	操作辅助rehash:在redis中每一个增删改查命令中都会判断数据库字典中的哈希表是否正在进行渐进式rehash，如果是则帮助执行一次
        	定时辅助rehash:虽然redis实现了在读写操作时，辅助服务器进行渐进式rehash操作，但是如果服务器比较空闲，redis数据库将很长时间内都一直使用两个哈希表。所以在redis周期函数中，如果发现有字典正在进行渐进式rehash操作，则会花费1毫秒的时间，帮助一起进行渐进式rehash操作
```

**Python代码hash散列**

```python

```

**应用场景：用户维度数据统计**

```python
用户维度统计
   统计数包括：关注数、粉丝数、喜欢商品数、发帖数
   用户为key，不同维度为field，value为统计数
   比如关注了5人
	 HSET user:10000 fans 5
	 HINCRBY user:10000 fans 1
```

**应用场景: 缓存 - redis+mysql+hash组合使用**

- **原理**

  ```python
  用户想要查询个人信息
  1、到redis缓存中查询个人信息
  2、redis中查询不到，到mysql查询，并缓存到redis
  3、再次查询个人信息
  ```

- **代码实现**

  ```python
  
  ```




## **集合数据类型（set）**

- **特点**

```python
1、无序、去重
2、元素是字符串类型
3、最多包含2^32-1个元素
```

- **基本命令**

```python
# 1、增加一个或者多个元素,自动去重；返回值为成功插入到集合的元素个数
SADD key member1 member2
# 2、查看集合中所有元素
SMEMBERS key
# 3、删除一个或者多个元素，元素不存在自动忽略
SREM key member1 member2
# 4、元素是否存在
SISMEMBER key member
# 5、随机返回集合中指定个数的元素，默认为1个
SRANDMEMBER key [count]
# 6、弹出成员
SPOP key [count]
# 7、返回集合中元素的个数，不会遍历整个集合，只是存储在键当中了
SCARD key
# 8、把元素80202 222258022从源集合移动到目标集合
SMOVE source destination member

# 9、差集(number1 1 2 3 number2 1 2 4 结果为3)
SDIFF key1 key2 
# 10、差集保存到另一个集合中
SDIFFSTORE destination key1 key2

# 11、交集
SINTER key1 key2
SINTERSTORE destination key1 key2

# 11、并集
SUNION key1 key2
SUNIONSTORE destination key1 key2
```

**案例: 新浪微博的共同关注**

```python
# 需求: 当用户访问另一个用户的时候，会显示出两个用户共同关注过哪些相同的用户
# 设计: 将每个用户关注的用户放在集合中，求交集即可
# 实现:
	user001 = {'peiqi','qiaozhi','danni'}
	user002 = {'peiqi','qiaozhi','lingyang'}
  
user001和user002的共同关注为:
	SINTER user001 user002
	结果为: {'peiqi','qiaozhi'}
```

**python操作set**

```python
r. sadd('pyset1', 'tom', 'jack')
print(r . smembers(' pyset1'))
```



## **有序集合sortedset**

- **特点**

```
1、有序、去重
2、元素是字符串类型
3、每个元素都关联着一个浮点数分值(score)，并按照分值从小到大的顺序排列集合中的元素（分值可以相同）
4、最多包含2^32-1元素
```

- 示例

  **一个保存了水果价格的有序集合**

| 分值 | 2.0  | 4.0  | 6.0  | 8.0  | 10.0 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 元素 | 西瓜 | 葡萄 | 芒果 | 香蕉 | 苹果 |

	**一个保存了员工薪水的有序集合**

| 分值 | 6000 | 8000 | 10000 | 12000 |      |
| ---- | ---- | ---- | ----- | ----- | ---- |
| 元素 | lucy | tom  | jim   | jack  |      |

	**一个保存了正在阅读某些技术书的人数**

| 分值 | 300      | 400    | 555    | 666        | 777      |
| ---- | -------- | ------ | ------ | ---------- | -------- |
| 元素 | 核心编程 | 阿凡提 | 本拉登 | 阿姆斯特朗 | 比尔盖茨 |

- **有序集合常用命令**

```python
# 在有序集合中添加一个成员 返回值为 成功插入到集合中的元素个数
zadd key score member
# 查看指定区间元素（升序)
zrange key start stop [withscores]
# 查看指定区间元素（降序）
zrevrange key start stop [withscores]
# 查看指定元素的分值
zscore key member

# 返回指定区间元素
# offset : 跳过多少个元素
# count : 返回几个
# 小括号 : 开区间  zrangebyscore fruits (2.0 8.0
zrangebyscore key min max [withscores] [limit offset count]
# 每页显示10个成员,显示第5页的成员信息:
# limit 40 10
# MySQL: 每页显示10条记录,显示第5页的记录
# limit 40,10
# limit 2,3   显示: 第3 4 5条记录

# 删除成员
zrem key member
# 增加或者减少分值
zincrby key increment member
# 返回元素排名
zrank key member
# 返回元素逆序排名
zrevrank key member
# 删除指定区间内的元素
zremrangebyscore key min max
# 返回集合中元素个数
zcard key
# 返回指定范围中元素的个数
zcount key min max
zcount salary 6000 8000
zcount salary (6000 8000# 6000<salary<=8000
zcount salary (6000 (8000#6000<salary<8000             
# 并集
zunionstore destination numkeys key [weights 权重值] [AGGREGATE SUM|MIN|MAX]
# zunionstore salary3 2 salary salary2 weights 1 0.5 AGGREGATE MAX
# 2代表集合数量,weights之后 权重1给salary,权重0.5给salary2集合,算完权重之后执行聚合AGGREGATE
                     
# 交集：和并集类似，只取相同的元素
zinterstore destination numkeys key1 key2 weights weight AGGREGATE SUM(默认)|MIN|MAX
```

**python操作sorted set**

```python
r.zadd('pyss1', {'tom' : 6000, 'jim' :5000})
# 分数是浮点型
print(r.zrange('pyss1' ,0,-1,withscores=True))
print(r.zrangebyscore('pyss1', '(5000', 5500, start=1, num=2, Withscores=True))
r.zinterstore('pyss3',('pyss1' ,'pyss2'),aggregate= 'max')
print(r .zrange('pyss3', 0，-1,withscores=True))
```





# redis_day02回顾

## **五大数据类型及应用场景**

| 类型       | 特点                                                         | 使用场景                                                     |
| :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| string     | 简单key-value类型，value可为字符串和数字                     | 常规计数（微博数, 粉丝数等功能）                             |
| hash       | 是一个string类型的field和value的映射表，hash特别适合用于存储对象 | 存储部分可能需要变更的数据（比如用户信息）                   |
| list       | 有序可重复列表                                               | 消息队列等                                                   |
| set        | 无序不可重复列表                                             | 存储并计算关系（如微博，关注人或粉丝存放在集合，可通过交集、并集、差集等操作实现如共同关注、共同喜好等功能） |
| sorted set | 每个元素带有分值的集合                                       | 各种排行榜                                                   |

# **redis_day03笔记**

### 事务

基本的四大特性

原子性(Atomicity) :事务中的操作,要么全都执行成功，要么全都不执行

一致性(Consistency): 事务执行始得数据从一个状态转换为另-个状态,这个过程保持完整性

隔离性(Isolation):当多个并发访问数据库是，多个事务之间相互隔离,不能被其他事务操作所干扰

持久性(Durability) :事务完成后，对于数据的改变是永久的

**redis的特点**

```python
1. 单独的隔离操作：事务中的所有命令会被序列化、按顺序执行，在执行的过程中不会被其他客户端发送来的命令打断
2. 不保证原子性：redis中的一个事务中如果存在命令执行失败，那么其他命令依然会被执行，没有回滚机制
3.触发事务之后，redis会把事务内的所有命令储存在队列中（数组），直到收到执行命令之后全部执行。
```

**事务命令**

```python
1、MULTI  # 开启事务          mysql   begin
2、命令1  # 执行命令          
3、命令2 ... ...
4、EXEC  # 提交到数据库执行    mysql   commit
5、DISCARD # 取消事务         mysql  'rollback'  取消不代表回滚
```

**使用步骤**

```python
# 开启事务
127.0.0.1:6379> MULTI
OK
# 命令1入队列
127.0.0.1:6379> INCR n1
QUEUED
# 命令2入队列
127.0.0.1:6379> INCR n2
QUEUED
# 提交到数据库执行
127.0.0.1:6379> EXEC
1) (integer) 1
2) (integer) 1
```

**事务中命令错误处理**

```python
# 1、命令语法错误，命令入队失败，直接自动discard退出这个事务
  这个在命令在执行调用之前会发生错误。例如，这个命令可能有语法错误（错误的参数数量，错误的命令名）
  处理方案：语法错误则自动执行discard

案例：
127.0.0.1:6379[7]> MULTI
OK
127.0.0.1:6379[7]> get a
QUEUED
127.0.0.1:6379[7]> getsss a
(error) ERR unknown command 'getsss'
127.0.0.1:6379[7]> 
127.0.0.1:6379[7]> 
127.0.0.1:6379[7]> EXEC
(error) EXECABORT Transaction discarded because of previous errors.

# 2、命令语法没错，但类型操作有误，则事务执行调用之后失败，无法进行事务回滚
   我们执行了一个由于错误的value的key操作（例如对着String类型的value施行了List命令操作） 
   处理方案：发生在EXEC之后的是没有特殊方式去处理的：即使某些命令在事务中失败，其他命令都将会被执行。

案例
127.0.0.1:6379> MULTI
OK
127.0.0.1:6379> set num 10
QUEUED
127.0.0.1:6379> LPOP num
QUEUED
127.0.0.1:6379> exec
1) OK
2) (error) WRONGTYPE Operation against a key holding the wrong kind of value
127.0.0.1:6379> get num
"10"
127.0.0.1:6379> 
```

**思考为什么redis不支持回滚？**



#### pipeline 流水线

定义：批量执行redis命令，减少通信io

注意：此为客户端技术

示例

```python
import redis
# 创建连接池并连接到redis    命令和命令之间如果有耦合关系，就不适合用池子
pool = redis.ConnectionPool(host = '127.0.0.1',db=0,port=6379)
r = redis.Redis(connection_pool=pool)

pipe = r.pipeline()
pipe.set('fans',50)
pipe.incr('fans')
pipe.incrby('fans',100)
pipe.execute()
```

性能对比

```python
# 创建连接池并连接到redis   
pool = redis.ConnectionPool(host = '127.0.0.1',db=0,port=6379)
r = redis.Redis(connection_pool=pool)
#  节省了网络通信次数之后带来的时间优化
def withpipeline(r):
    p = r.pipeline()
    for i in range(1000):
        key = 'test1' + str(i)
        value = i+1
        p.set(key, value)
    p.execute()

def withoutpipeline(r):
    for i in range(1000):
        key = 'test2' + str(i)
        value = i+1
        r.set(key, value)
```

**python 操作 redis事务**

```python
# 推介用法，参数transaction默认为true
with r.pipeline(transaction=True) as pipe
    pipe.multi()
    pipe.incr("books")
    pipe.incr("books")T
    values = pipe.execute()
# redis收到的指令（字符串）："multi incr books incr books exec"
```

#### watch - 乐观锁

作用： 事务过程中，可对指定key进行监听，命令提交时，若被监听key对应的值未被修改时，事务方可提交成功，否则失败

```python
开启事务之前使用 watch
> watch books
OK
> multi
OK
> incr books
QUEUED
> exec  # 事务执行失败
(nil)


watch之后，再开一个终端进入redis
> incr books  # 修改book值
(integer) 1


```

**python操作watch**

```python
#同时对一个账户进行操作， 当前余额 * 2
pyredis如果乐观锁返回None,则会直接报错，需要使用try来捕捉
	except redis.WatchError:

pipe.watch(key)  # 这个命令是直接发给redis了，其他都是打包
```



### 数据持久化

**持久化定义**

```python
将数据从掉电易失的内存放到永久存储的设备上
```

**为什么需要持久化**

```python
因为所有的数据都在内存上，所以必须得持久化
```

#### RDB模式（默认开启）

```python
1、保存真实的数据
2、将服务器包含的所有数据库数据以二进制文件的形式保存到硬盘里面
3、默认文件名(ubuntu) ：/var/lib/redis/dump.rdb
```

**创建rdb文件的两种方式**

**方式一：**redis终端中使用SAVE或者BGSAVE命令

```python
127.0.0.1:6379> SAVE
OK
# 特点
1、执行SAVE命令过程中，redis服务器将被阻塞，无法处理客户端发送的命令请求，在SAVE命令执行完毕后，服务器才会重新开始处理客户端发送的命令请求
2、如果RDB文件已经存在，那么服务器将自动使用新的RDB文件代替旧的RDB文件
# 工作中定时持久化保存一个文件

127.0.0.1:6379> BGSAVE
Background saving started
# 执行过程如下
1、客户端 发送 BGSAVE 给服务器
2、服务器马上返回 Background saving started 给客户端
3、服务器 fork() 子进程做这件事情
4、服务器继续提供服务
5、子进程创建完RDB文件后再告知Redis服务器

# 配置文件相关
/etc/redis/redis.conf
263行: dir /var/lib/redis # 表示rdb文件存放路径
253行: dbfilename dump.rdb  # 文件名   使用多台redis时需要修改

# 两个命令比较
SAVE比BGSAVE快，因为需要创建子进程，消耗额外的内存

# 补充：可以通过查看日志文件来查看redis都做了哪些操作
# 日志文件：配置文件中搜索 logfile
logfile /var/log/redis/redis-server.log
```

**方式二：**设置配置文件条件满足时自动保存**（使用最多）**

```python
# redis配置文件默认
218行: save 900 1
219行: save 300 10
220行: save 60 10000
    表示如果距离上一次创建RDB文件已经过去了300秒，并且服务器的所有数据库总共已经发生了不少于10次修改，那么自动执行BGSAVE命令
  1、只要三个条件中的任意一个被满足时，服务器就会自动执行BGSAVE
  2、每次创建RDB文件之后，服务器为实现自动持久化而设置的时间计数器和次数计数器就会被清零，并重新开始计数，所以多个保存条件的效果不会叠加
    
# 该配置项也可以在命令行执行 [不推荐] 
redis>save 60 10000
```

**RDB缺点**

```shell
1、创建RDB文件需要将服务器所有的数据库的数据都保存起来，这是一个非常消耗资源和时间的操作，所以服务器需要隔一段时间才创建一个新的RDB文件，也就是说，创建RDB文件不能执行的过于频繁，否则会严重影响服务器的性能
2、可能丢失数据
```



#### AOF（AppendOnlyFile）

```python
1、存储的是命令，而不是真实数据
2、默认不开启
# 开启方式（修改配置文件）
1、/etc/redis/redis.conf
  672行: appendonly yes # 把 no 改为 yes
  676行: appendfilename "appendonly.aof"
2、重启服务
  sudo /etc/init.d/redis-server restart
```

**AOF持久化原理及优点**

```python
# 原理
   1、每当有修改数据库的命令被执行时，
   2、因为AOF文件里面存储了服务器执行过的所有数据库修改的命令，所以给定一个AOF文件，服务器只要重新执行一遍AOF文件里面包含的所有命令，就可以达到还原数据库的目的

# 优点
  用户可以根据自己的需要对AOF持久化进行调整，让Redis在遭遇意外停机时不丢失任何数据，或者只丢失一秒钟的数据，这比RDB持久化丢失的数据要少的多
```

**特殊说明**

```python
# 因为
  虽然服务器执行一个修改数据库的命令，就会把执行的命令写入到AOF文件，但这并不意味着AOF文件持久化不会丢失任何数据，在目前常见的操作系统中，执行系统调用write函数，将一些内容写入到某个文件里面时，为了提高效率，系统通常不会直接将内容写入硬盘里面，而是将内容放入一个内存缓存区（buffer）里面，等到缓冲区被填满时才将存储在缓冲区里面的内容真正写入到硬盘里

# 所以
  1、AOF持久化：当一条命令真正的被写入到硬盘里面时，这条命令才不会因为停机而意外丢失
  2、AOF持久化在遭遇停机时丢失命令的数量，取决于命令被写入到硬盘的时间
  3、越早将命令写入到硬盘，发生意外停机时丢失的数据就越少，反之亦然
```

```

```

**策略 - 配置文件**

```python
# 打开配置文件:/etc/redis/redis.conf，找到相关策略如下
1、701行: always
   服务器每写入一条命令，就将缓冲区里面的命令写入到硬盘里面，服务器就算意外停机，也不会丢失任何已经成功执行的命令数据
2、702行: everysec（# 默认）
   服务器每一秒将缓冲区里面的命令写入到硬盘里面，这种模式下，服务器即使遭遇意外停机，最多只丢失1秒的数据
3、703行: no
   服务器不主动将命令写入硬盘,由操作系统决定何时将缓冲区里面的命令写入到硬盘里面，丢失命令数量不确定

# 运行速度比较
always：速度慢
everysec和no都很快，默认值为everysec
```

**AOF重写**

**思考：AOF文件中是否会产生很多的冗余命令？**

```python
为了让AOF文件的大小控制在合理范围，避免胡乱增长，redis提供了AOF重写功能，通过这个功能，服务器可以产生一个新的AOF文件
  -- 新的AOF文件记录的数据库数据和原由的AOF文件记录的数据库数据完全一样
  -- 新的AOF文件会使用尽可能少的命令来记录数据库数据，因此新的AOF文件的提及通常会小很多
  -- AOF重写期间，服务器不会被阻塞，可以正常处理客户端发送的命令请求
```

**示例**

| 原有AOF文件                | 重写后的AOF文件                         |
| -------------------------- | --------------------------------------- |
| select 0                   | SELECT 0                                |
| sadd myset peiqi           | SADD myset peiqi qiaozhi danni lingyang |
| sadd myset qiaozhi         | SET msg 'hello tarena'                  |
| sadd myset danni           | RPUSH mylist 2 3 5                      |
| sadd myset lingyang        |                                         |
| INCR number                |                                         |
| INCR number                |                                         |
| DEL number                 |                                         |
| SET message 'hello world'  |                                         |
| SET message 'hello tarena' |                                         |
| RPUSH mylist 1 2 3         |                                         |
| RPUSH mylist 5             |                                         |
| LPOP mylist                |                                         |

**AOF重写-触发**

```python
1、客户端向服务器发送BGREWRITEAOF命令
   127.0.0.1:6379> BGREWRITEAOF
   Background append only file rewriting started

2、修改配置文件让服务器自动执行BGREWRITEAOF命令
  auto-aof-rewrite-percentage 100
  auto-aof-rewrite-min-size 64mb
  # 解释
    1、只有当AOF文件的增量大于100%时才进行重写，也就是大一倍的时候才触发
        # 第一次重写新增：64M
        # 第二次重写新增：128M
        # 第三次重写新增：256M（新增128M）
```



**RDB和AOF持久化对比**

| RDB持久化                                                    | AOF持久化                                     |
| ------------------------------------------------------------ | --------------------------------------------- |
| 全量备份，一次保存整个数据库                                 | 增量备份，一次保存一个修改数据库的命令        |
| 保存的间隔较长                                               | 保存的间隔默认为一秒钟                        |
| 数据还原速度快                                               | 数据还原速度一般，冗余命令多，还原速度慢      |
| 执行SAVE命令时会阻塞服务器，但手动或者自动触发的BGSAVE不会阻塞服务器 | 无论是平时还是进行AOF重写时，都不会阻塞服务器 |
|                                                              |                                               |

```python
# 用redis用来存储真正数据，每一条都不能丢失，都要用always，有的做缓存，有的保存真数据，我可以开多个redis服务，不同业务使用不同的持久化，新浪每个服务器上有4个redis服务，整个业务中有上千个redis服务，分不同的业务，每个持久化的级别都是不一样的。
```

**数据恢复（无需手动操作）**

```python
既有dump.rdb，又有appendonly.aof，恢复时找谁？
先找appendonly.aof
```

**配置文件常用配置总结**

```python
# 设置密码
1、requirepass password
# 开启远程连接
2、bind 127.0.0.1 ::1 注释掉
3、protected-mode no  把默认的 yes 改为 no
# rdb持久化-默认配置
4、dbfilename 'dump.rdb'
5、dir /var/lib/redis
# rdb持久化-自动触发(条件)
6、save 900 1
7、save 300 10 
8、save 60  10000
# aof持久化开启
9、appendonly yes
10、appendfilename 'appendonly.aof'
# aof持久化策略
11、appendfsync always
12、appendfsync everysec # 默认
13、appendfsync no
# aof重写触发
14、auto-aof-rewrite-percentage 100
15、auto-aof-rewrite-min-size 64mb
# 设置为从服务器
16、salveof <master-ip> <master-port>
```

**Redis相关文件存放路径**

```python
1、配置文件: /etc/redis/redis.conf
2、备份文件: /var/lib/redis/*.rdb|*.aof
3、日志文件: /var/log/redis/redis-server.log
4、启动文件: /etc/init.d/redis-server
# /etc/下存放配置文件
# /etc/init.d/下存放服务启动文件
```



## **Redis主从复制**

- **定义**

```python
1、一个Redis服务可以有多个该服务的复制品，这个Redis服务成为master，其他复制品成为slaves
2、master会一直将自己的数据更新同步给slaves，保持主从同步
3、只有master可以执行写命令，slave只能执行读命令
```

- **作用**

```python
1.分担了读的压力（高并发）  整体提高了redis的服务能力
2.避免单点问题【如果系统中，一个进程挂掉，整个系统就挂掉了】
```

- **原理**

```python
从服务器执行客户端发送的读命令，比如GET、LRANGE、SMEMMBERS、HGET、ZRANGE等等，客户端可以连接slaves执行读请求，来降低master的读压力
```

- **实现方式**
  - **方式一**（Linux命令行实现）

    redis-server --slaveof <master-ip> <master-port> --masterauth <master password>

    （两个 - -）
  
    ```shell
    # 从服务端
    redis-server --port 6300 --slaveof 127.0.0.1 6379
    # 从客户端
    redis-cli -p 6300
    127.0.0.1:6300> keys * 
    # 发现是复制了原6379端口的redis中数据
    127.0.0.1:6300> set mykey 123
    (error) READONLY You can't write against a read only slave.
    127.0.0.1:6300> 
    ```
  # 从服务器只能读数据，不能写数据
  ```
  
  - **方式二**（Redis命令行实现）
  
    ```shell
    # 两条命令
    1、>slaveof IP PORT
    2、>slaveof no one
    
    # 服务端启动
    redis-server --port 6301
    # 客户端连接
    tarena@tedu:~$ redis-cli -p 6301
    127.0.0.1:6301> keys *
    1) "myset"
    2) "mylist"
    127.0.0.1:6301> set mykey 123
    OK
    # 切换为从
    127.0.0.1:6301> slaveof 127.0.0.1 6379
    OK
    127.0.0.1:6301> set newkey 456
    (error) READONLY You can't write against a read only slave.
    127.0.0.1:6301> keys *
    1) "myset"
    2) "mylist" 
    # 再切换为主
    127.0.0.1:6301> slaveof no one
    OK
    127.0.0.1:6301> set name hello
    OK
  
  ```
  
  - **方式三**(利用配置文件)
  
    ```shell
    # 每个redis服务,都有1个和他对应的配置文件
    # 两个redis服务
      1、6379 -> /etc/redis/redis.conf
      2、6300 -> /home/tarena/redis_6300.conf
    
    # 修改配置文件
    vi redis_6300.conf
    slaveof 127.0.0.1 6379
    port 6300
    # 启动redis服务
    redis-server redis_6300.conf
    # 客户端连接测试
    redis-cli -p 6300
    127.0.0.1:6300> hset user:1 username guods
    (error) READONLY You can't write against a read only slave.
    ```

**问题：master挂了怎么办？**

```python
1、一个Master可以有多个Slaves
2、Slave下线，只是读请求的处理性能下降
3、Master下线，写请求无法执行
4、其中一台Slave使用SLAVEOF no one命令成为Master，其他Slaves执行SLAVEOF命令指向这个新的Master，从它这里同步数据
# 以上过程是手动的，能够实现自动，这就需要Sentinel哨兵，实现故障转移Failover操作
```

**演示**

```python
1、启动端口6400redis，设置为6379的slave
   redis-server --port 6400
   redis-cli -p 6400
   redis>slaveof 127.0.0.1 6379
2、启动端口6401redis，设置为6379的slave
   redis-server --port 6401
   redis-cli -p 6401
   redis>slaveof 127.0.0.1 6379
3、关闭6379redis
   sudo /etc/init.d/redis-server stop
4、把6400redis设置为master
   redis-cli -p 6400
   redis>slaveof no one
5、把6401的redis设置为6400redis的salve
   redis-cli -p 6401
   redis>slaveof 127.0.0.1 6400
# 这是手动操作，效率低，而且需要时间，有没有自动的？？？
```

## **Sentinel哨兵**

**Redis之哨兵 - sentinel**

```python
1、Sentinel会不断检查Master和Slaves是否正常
2、每一个Sentinel可以监控任意多个Master和该Master下的Slaves
```

**案例演示**

	**1、**环境搭建

```python
# 共3个redis的服务
1、启动6379的redis服务器
   	sudo /etc/init.d/redis-server start
2、启动6380的redis服务器，设置为6379的从
    redis-server --port 6380
    tarena@tedu:~$ redis-cli -p 6380
    127.0.0.1:6380> slaveof 127.0.0.1 6379
    OK
3、启动6381的redis服务器，设置为6379的从
   	redis-server --port 6381
   	tarena@tedu:~$ redis-cli -p 6381
   	127.0.0.1:6381> slaveof 127.0.0.1 6379
```

	**2、**安装并搭建sentinel哨兵

```python
# 1、安装redis-sentinel
sudo apt install redis-sentinel
验证: sudo /etc/init.d/redis-sentinel stop
# 2、新建配置文件sentinel.conf
port 26379
sentinel monitor tedu 127.0.0.1 6379 1
# 3、启动sentinel
方式一: redis-sentinel sentinel.conf
方式二: redis-server sentinel.conf --sentinel
#4、将master的redis服务终止，查看从是否会提升为主
sudo /etc/init.d/redis-server stop
# 发现提升6381为master，其他两个为从
# 在6381上设置新值，6380查看
127.0.0.1:6381> set name tedu
OK

# 启动6379，观察日志，发现变为了6381的从
主从+哨兵基本就够用了
```

sentinel.conf解释

```python
# sentinel监听端口，默认是26379，可以修改
port 26379
# 告诉sentinel去监听地址为ip:port的一个master，这里的master-name可以自定义，quorum是一个数字，指明当有多少个sentinel认为一个master失效时，master才算真正失效
sentinel monitor <master-name> <ip> <redis-port> <quorum>

#如果master有密码，则需要添加该配置
sentinel auth-pass <master-name> <password>

#master多久失联才认为是不可用了，默认是30秒
sentinel down-after-milliseconds <master-name> <milliseconds> 
```



**python获取master**

```python
from redis.sentinel import Sentinel

#生成哨兵连接
sentinel = Sentinel([('localhost', 26379),(…………)], socket_timeout=0.1)
socket_timeout   连接超时时间，超过就报错，可以不配置

#初始化master连接
master = sentinel.master_for('tedu', socket_timeout=0.1, db=1)
slave = sentinel.slave_for('tedu',socket_timeout=0.1, db=1)

#使用redis相关命令
master.set('mymaster', 'yes')
print(slave.get('mymaster'))
```

分布式集群服务：