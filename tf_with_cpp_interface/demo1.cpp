//需包含头文件，和两个namespace声明

# include "scope.h"
using namespace tensorflow;
using namespace ops;

//该类的主要成员（我们当然只关心调用pb文件完成预测的任务，至于读取图的某一个节点，自然是不关心的，至少在C++端不会在意这个，那是用Python该做的事情），我们主要会使用到的成员函数有：

Scope::NewRootScope(); 创建一个根空间，意味着所有的张量的名字都有一个前缀，根空间的前缀是空的；
Scope::NewSubScope(const &string);

//下面看两个例子：
//头文件我设置为：

#pragma once
#define COMPILER_MSVC
#define NOMINMAX

#include "scope.h"
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace ops;

void testScope();
void testScopeDemo();

//源文件如下

#include "scope.h"

void testScope()
{
	Scope root = Scope::NewRootScope();
	auto c1 = Const(root, { { 1,1 } });
	auto m = MatMul(root, c1, { { 41 },{ 1 } });
	GraphDef gdef;
	Status s = root.ToGraphDef(&gdef);
	if (!s.ok()) {
		std::cout << "error" << std::endl;
	}
	else {
		std::cout << "successfully generate graph!" << std::endl;
	}
}
void testScopeDemo()
{
	Scope root = Scope::NewRootScope();
	Scope linear = root.NewSubScope("linear");
	// W will be named "linear/W"
	auto W = Variable(linear.WithOpName("W"), { 2,2 },DT_FLOAT);
	// b will be named "linear/b"
	auto b = Variable(linear.WithOpName("b"), { 2,2 }, DT_FLOAT);
	auto x = Const(linear, { 1 });
	auto m = MatMul(linear, x, W);
	auto r = BiasAdd(linear,m, b);
}

//在第一个函数testScope中，先创建一个root空间。之后所有的operation，第一个参数都是这个root变量。
//在第二个函数testScopeDemo中，先创建一个root空间，在创建一个子空间，叫做‘linear’。在linear之下创建的所有张量，其名称的前缀都是“linear/”。包括Variable和Const生成的张量，以及operation操作输出的张量。
//————————————————
//版权声明：本文为CSDN博主「咆哮的阿杰」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/qq_34914551/article/details/93795450

//TensorFlow的C++之路（2）：导入Pb模型
//
//咆哮的阿杰 2019-07-02 09:45:54   878   收藏 2
//展开
//上一篇博客讲到了scope，在TF的C++API中，所有的operation的第一个参数都是scope。其实C++的tf api有很多不一样的特性。我们往后慢慢学习，慢慢总结。
//
//OK，这次就写篇实际的，也是这两天很折腾人的环节。我们得到了Pb文件，想在C++端使用，以便后续封装生成dll。那么如何导入Pb模型呢？TF还是很友好的，提供了ReadBinaryProto导入pb模型。
//
//这里我写了一个函数，大家可以直接拿去用：

Status readPb(Session  *sess, GraphDef &gdef, const string &modelPath)
{
	Status status = ReadBinaryProto(Env::Default(), modelPath, &gdef);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
	}
	else
	{
		cout << "load graph protobuf successfully" << std::endl;
	}
	status = sess->Create(gdef);
	return status;

}
//需要包含以下的头文件：

#include "tensorflow\core\public\session.h"
#include "tensorflow\core\platform\env.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using std::cin;
using std::cout;
using std::vector;
//自定义的readPb函数需要一个session类型的指针，一个GraphDef的引用类型，还有string类型，内容是pb文件的地址。

//下面我们看看主函数怎么写：

Session * session = 0;
Status status = NewSession(SessionOptions(), &session);
Scope root = Scope::NewRootScope();
GraphDef gdef;
status = readPb(session, gdef, "flowNet.pb");
if (!status.ok()) {
	std::cerr << status.ToString() << std::endl;
}
//这样就可以调用pb文件了。Status类型可以帮助我们检测操作是否成功，如果成功，Status的成员函数ok()为True。如果操作不成功，我们可以使用ToString函数成员来输出未能导入图的问题。

//常常是sess->Create这里会有问题。比如我就遇到过提示LeakyRelu这个节点找不到，原来是因为python中支持的tf.nn.leaky_relu在C++的API中不存在。遇到这种情况，只能去把python的leaky_relu换成使用普通函数写成的等价形式。
————————————————
//版权声明：本文为CSDN博主「咆哮的阿杰」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/qq_34914551/article/details/94213216