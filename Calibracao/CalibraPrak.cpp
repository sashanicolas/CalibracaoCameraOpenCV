/*
 * Implementecao Inicial de calibracao de camera com padrao circular
 * Sasha Nicolas
 * Tecgraf PUC Rio
 */

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

// variaveis
vector<string> imagePaths;
vector<Mat> originalImages;
vector<vector<Point2d>> pontosDoCanto;
int quantPontosControleWidth, quantPontosControleHeight;
int countPoints;
vector<vector<vector<Point2d>>> pontosDoGridDeCadaImagem;

// prototipos
void carregaImagens();
void onMouseCallBack(int e, int x, int y, int flags, void * param);
void desenhaCruz(Mat aux, int x, int y, Scalar color);
void desenhaGrid(Mat img, vector<Point2d> pontos, int id);
void mostraImagem(Mat img, string name);

int main(){
	//carregar definicoes
	quantPontosControleWidth = 9;
	quantPontosControleHeight = 6;

	// Selecionar Imagens
	carregaImagens();

	// Selecionar 4 pontos limite do padrao para cada foto
	Mat aux;
	for (int i = 0; i < imagePaths.size(); i++){
		countPoints = 0;
		mostraImagem(originalImages[i], imagePaths[i]);
		setMouseCallback(imagePaths[i], onMouseCallBack, &i);

		while (countPoints<4){
			waitKey(20);
		}

		setMouseCallback(imagePaths[i], NULL, NULL);
		
		//desenhar o grid	
		aux = originalImages[i].clone();
		desenhaGrid(aux, pontosDoCanto[i], i);
		mostraImagem(aux, imagePaths[i]);
		waitKey(0);

		destroyWindow(imagePaths[i]);		
	}

	return 0;
}

void carregaImagens(){
	printf("Carregando Imagens\n");
	imagePaths.push_back("imagens/circle1.jpg");
	imagePaths.push_back("imagens/circle2.jpg");
	/*imagePaths.push_back("imagens/circle3.jpg");
	imagePaths.push_back("imagens/circle4.jpg");
	imagePaths.push_back("imagens/circle5.jpg");*/

	for (int i = 0; i < imagePaths.size(); i++){
		Mat aux, aux2;
		aux = imread(imagePaths[i]);
		resize(aux, aux, Size(aux.size().width / 4, aux.size().height / 4));
		originalImages.push_back(aux);
	}
	printf("Feito!\n");
}

void onMouseCallBack(int e, int x, int y, int flags, void * imageId){
	if (e == CV_EVENT_LBUTTONDOWN){
		int id = *((int*)imageId);
		printf("x,y (%d,%d)\n", x, y);

		Point2d ponto(x, y);

		if (id >= pontosDoCanto.size()){
			vector<Point2d> v;
			pontosDoCanto.push_back(v);
		}
		pontosDoCanto[id].push_back(ponto);

		// colocar marca no ponto clicado
		Mat aux = originalImages[id].clone();
		for (int i = 0; i < pontosDoCanto[id].size(); i++){
			int x = pontosDoCanto[id][i].x, y = pontosDoCanto[id][i].y;
			desenhaCruz(aux, x, y, Scalar(0,0,255));
		}

		mostraImagem(aux, imagePaths[id]);

		countPoints++;
	}
}

void desenhaCruz(Mat aux, int x, int y, Scalar color){
	line(aux, Point(x - 5, y), Point(x + 5, y), color, 1, 8);
	line(aux, Point(x, y - 5), Point(x, y + 5), color, 1, 8);
}

void desenhaLinha(Mat aux, Point2d a, Point2d b, Scalar color){
	line(aux, a, b, color, 1, 8);
}

void desenhaGrid(Mat img, vector<Point2d> pontosCanto, int id){
	/*for (int i = 0; i < 4; i++){
		int x = pontosDoCanto[id][i].x, y = pontosDoCanto[id][i].y;
		desenhaCruz(img, x, y, Scalar(0, 0, 255));
	}*/
	//marca os pontos dos lados da figura quadrilatera
	vector<vector<Point2d>> pontosBorda;
	for (int i = 0; i < 4; i++){
		int quant = (i % 2 == 0) ? quantPontosControleWidth : quantPontosControleHeight;
		float delta = norm(pontosCanto[i == 3 ? 0: i + 1] - pontosCanto[i]) / (float)quant;

		Point2d direcao = pontosCanto[i == 3 ? 0 : i + 1] - pontosCanto[i];
		direcao *= 1 / cv::norm(direcao);
		Point2d p = pontosCanto[i];

		vector<Point2d> pontosLado;
		for (int j = 0; j < quant - 1; j++){
			p += direcao*delta;
			desenhaCruz(img, p.x, p.y, Scalar(0, 255, 0));
			pontosLado.push_back(p);
		}
		pontosBorda.push_back(pontosLado);
	}
	if (id >= pontosDoGridDeCadaImagem.size()){		
		pontosDoGridDeCadaImagem.push_back(pontosBorda);
	}
	else
		pontosDoGridDeCadaImagem[id] = pontosBorda;
	
	//desenha linhas do grid
	//linhas externas
	for (int i = 0; i < 4; i++){
		desenhaLinha(img, pontosCanto[i], pontosCanto[i == 3 ? 0 : i + 1], Scalar(255,0,0));
	}	
	//linhas verticais
	for (int i = 0; i < quantPontosControleWidth-1; i++){
		desenhaLinha(img, pontosBorda[0][i], pontosBorda[2][quantPontosControleWidth - i - 2], Scalar(255, 0, 0));
	}
	//linhas horizontais
	for (int i = 0; i < quantPontosControleHeight - 1; i++){
		desenhaLinha(img, pontosBorda[1][i], pontosBorda[3][quantPontosControleHeight - i - 2], Scalar(255, 0, 0));
	}
}

void mostraImagem(Mat img, string name){
	namedWindow(name);
	imshow(name, img);
	moveWindow(name, 150, 150);
}