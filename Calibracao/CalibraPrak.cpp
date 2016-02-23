/*
 * Implementecao Inicial de calibracao de camera com padrao circular
 * Sasha Nicolas
 * Tecgraf PUC Rio
 *
 * READ.ME:
 1) Passa as imagens em carregaImagens() - hard coded mesmo
 2) Passa a configuracao dos padroes
 nHorizontal = 9;
 nVertical = 6;
 distanceCP = 35; //milimetros
 3) Configura os #define (ativa ou desativa)
 REDUZIR_IMAGENS - se for carregar imagens muito grande, reduz por 4
 SELECT_CORNERS_MOUSE - para selecionar com o mouse os 4 cantos de cada imagem
 MOSTRA_GRID - para visualizar o grid calculado
 MOSTRA_CADA_ROI - para visulizar cada circulo segmentado (nao ativa - eh meio chato)
 MOSTRA_POSICAO_PC - para mostra o centro dos pontos de controles (PC)
 MOSTRA_REPROJECAO - mostra comparacao do da reprojecao

 ** tudo tem waitkey(0) - entao passa com evento do teclado
 ** para selecionar os 4 cantos comeca do canto superior esquerdo e vai girando em ordem horaria
 */

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <math.h> 
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

//DEBUG e CONTROLE
#define USA_CIRCULOS_SASHA 0 // sasha = 1, ankur = 0
#define REDUZIR_IMAGENS 0 //reduz imagens celular
#define SELECT_CORNERS_MOUSE 0 
#define MOSTRA_GRID 0
#define MOSTRA_CADA_ROI 0
#define MOSTRA_POSICAO_PC 0
#define MOSTRA_REPROJECAO 0
#define MOSTRA_UNDISTORTED 0
#define MOSTRA_UNPROJECTED 0
#define MOSTRA_PERSPECTIVE_TRANSFORM 0
#define AS_ANKUR 0 //ankur = 1, prak =0
#define SHOW_NOVOS_CENTROS_SSD 1 //ankur
#define SHOW_NOVOS_CENTROS_ELLIPSE 0 //prak


// variaveis
vector<string> imagePaths;
vector<Mat> originalImages, undistortedImages, frontoParallelImages;
vector<vector<Point2d>> pontosDoCanto;
int nHorizontal, nVertical, countPoints, distanceCP;
vector<vector<Point2f>> pontosGrid;
vector<Mat> homografias;
vector<vector<Point2f>> centrosOriginal;
vector<vector<Point3f>> centrosIdealObjeto;
vector<vector<Point2f>> centrosUndistorted;
vector<vector<Point2f>> centrosFrontoParallel;
vector<vector<Point2f>> novosCentrosFrontoParallel;
vector<vector<Point2f>> centrosProjetados;
vector<vector<Point2f>> centrosDistorcidos;

Mat cameraMatrix, distCoeffs;
int flag = CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_ZERO_TANGENT_DIST;
vector<Mat> rvecs, tvecs;
float distanciaCentro;


// prototipos
void carregaImagens();
void onMouseCallBack(int e, int x, int y, int flags, void * param);
void desenhaCruz(Mat aux, int x, int y, Scalar color);
void desenhaLinha(Mat aux, Point2d a, Point2d b, Scalar color);
void imprimeMat3xn(Mat m, int s);
void calculaGridROI(Mat img, vector<Point2d> pontos, int id);
void mostraImagem(Mat img, string name);
void computeEllipse(Mat img, int idImg);
void cantosPredefinidos();
void selecionarCantosComMouse();
void desenhaQuadrilatero(Mat img, Point p[4], Scalar c);
void calcPosicoesIdeaisObjeto();
static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors);
void print(Mat mat, int prec);
void calibraCamera();
void undistortImages();
void frontoParalelo();
void findCentrosFrontoParalelo();
vector<Point2f> computeCorrelationSSD(Mat image, vector<Point2f>);
vector<Point2f> fitEllipse(Mat image, vector<Point2f>);
void desenhaCentros(Mat img, vector<Point2f> centros);
void proj_dist_centros();
void desenhaCentros(Mat img, vector<Point2f> centros, Scalar cor);
void showHistogram(Mat b_hist, int histSize);

int main(){

	//configurar definicoes
	if (USA_CIRCULOS_SASHA){
		nHorizontal = 9;
		nVertical = 6;
	}
	else{
		nHorizontal = 10;
		nVertical = 7;
	}
	distanceCP = 23; //milimetros , 35

	// Carregar Imagens
	carregaImagens();

	//Selecionar cantos
	if (SELECT_CORNERS_MOUSE)
		selecionarCantosComMouse();
	else
		cantosPredefinidos();

	Mat aux;

	// Primeira iteracao
	// calcula ponto inicial dos circulos (centro da elipse eh a media das coordenadas)
	for (int i = 0; i < imagePaths.size(); i++){
		aux = originalImages[i].clone();
		computeEllipse(aux, i);
	}

	if (MOSTRA_POSICAO_PC){
		for (int k = 0; k < originalImages.size(); k++){
			aux = originalImages[k].clone();			
			desenhaCentros(aux, centrosOriginal[k], Scalar(255, 255, 255));
			mostraImagem(aux, imagePaths[k]);
			moveWindow(imagePaths[k], 150, 150);
			waitKey(0);
			cv::destroyWindow(imagePaths[k]);
		}
	}

	// Calibracao inicial
	calibraCamera();
		
	for (int iter = 0; iter < 60; iter++){
		cout << "iteracao " << iter << endl;
		undistortImages();
		frontoParalelo();
		findCentrosFrontoParalelo();
		proj_dist_centros();
		calibraCamera();
		
		/*if (iter == 4) break;
		cout << "\nTerminado? s ou n";
		char a;
		cin >> a;
		if (a == 's') break;*/
	}
	
	cout << "\nTerminado.";
	char a;
	cin >> a;

	return 0;
}

void carregaImagens(){
	printf("Carregando Imagens\n");

	//minhas imagens, celular nexus 5
	if (USA_CIRCULOS_SASHA){
		imagePaths.push_back("imagens/circle1.jpg");
		imagePaths.push_back("imagens/circle2.jpg");
		imagePaths.push_back("imagens/circle3.jpg");
		imagePaths.push_back("imagens/circle4.jpg");
		imagePaths.push_back("imagens/circle8.jpg");
	}
	else{
		//imagens do Ankur
		imagePaths.push_back("imagens/img1.bmp");
		imagePaths.push_back("imagens/img2.bmp");
		imagePaths.push_back("imagens/img3.bmp");
		imagePaths.push_back("imagens/img4.bmp");
		imagePaths.push_back("imagens/img5.bmp");
	}

	for (int i = 0; i < imagePaths.size(); i++){
		printf(" - Imagem %d...\n", i);
		Mat aux, aux2;
		aux = imread(imagePaths[i]);
		if (REDUZIR_IMAGENS)
			resize(aux, aux, Size(aux.size().width / 4, aux.size().height / 4));
		originalImages.push_back(aux);
	}
	printf("Feito!\n");
}

void cantosPredefinidos(){
	Mat aux;
	vector<Point2d> v;

	//Imagens Sasha
	if (USA_CIRCULOS_SASHA){
		v.push_back(Point2d(234, 156));
		v.push_back(Point2d(646, 120));
		v.push_back(Point2d(665, 454));
		v.push_back(Point2d(220, 454));
	}
	else{
		//Imagens Ankur
		v.push_back(Point2d(198, 93));
		v.push_back(Point2d(833, 99));
		v.push_back(Point2d(839, 549));
		v.push_back(Point2d(186, 545));
	}
	

	pontosDoCanto.push_back(v);
	aux = originalImages[0].clone();
	calculaGridROI(aux, pontosDoCanto[0], 0);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[0]);
		waitKey(0);
		cv::destroyWindow(imagePaths[0]);
	}

	v.clear();
	//imagens sasha
	if (USA_CIRCULOS_SASHA){
		v.push_back(Point2d(184, 147));
		v.push_back(Point2d(594, 161));
		v.push_back(Point2d(642, 455));
		v.push_back(Point2d(202, 507));
	}
	else{
		//imagen ankur
		v.push_back(Point2d(258, 106));
		v.push_back(Point2d(808, 148));
		v.push_back(Point2d(801, 513));
		v.push_back(Point2d(255, 547));
	}

	pontosDoCanto.push_back(v);
	aux = originalImages[1].clone();
	calculaGridROI(aux, pontosDoCanto[1], 1);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[1]);
		waitKey(0);
		cv::destroyWindow(imagePaths[1]);
	}

	v.clear();
	//imagens sasha
	if (USA_CIRCULOS_SASHA){
		v.push_back(Point2d(177, 173));
		v.push_back(Point2d(592, 162));
		v.push_back(Point2d(608, 458));
		v.push_back(Point2d(196, 506));
	}
	//imagens ankur
	else{
		v.push_back(Point2d(261, 109));
		v.push_back(Point2d(828, 95));
		v.push_back(Point2d(803, 544));
		v.push_back(Point2d(247, 474));
	}

	pontosDoCanto.push_back(v);
	aux = originalImages[2].clone();
	calculaGridROI(aux, pontosDoCanto[2], 2);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[2]);
		waitKey(0);
		cv::destroyWindow(imagePaths[2]);
	}

	v.clear();
	//imagens sasha
	if (USA_CIRCULOS_SASHA){
		v.push_back(Point2d(184, 162));
		v.push_back(Point2d(595, 116));
		v.push_back(Point2d(610, 455));
		v.push_back(Point2d(205, 450));
	}
	//imagens ankur
	else{
		v.push_back(Point2d(244, 126));
		v.push_back(Point2d(867, 151));
		v.push_back(Point2d(814, 533));
		v.push_back(Point2d(264, 523));
	}

	pontosDoCanto.push_back(v);
	aux = originalImages[3].clone();
	calculaGridROI(aux, pontosDoCanto[3], 3);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[3]);
		waitKey(0);
		cv::destroyWindow(imagePaths[3]);
	}

	v.clear();
	//imagens sasha
	if (USA_CIRCULOS_SASHA){
		v.push_back(Point2d(149, 134));
		v.push_back(Point2d(644, 96));
		v.push_back(Point2d(718, 483));
		v.push_back(Point2d(124, 512));
	}
	//imagens ankur
	else{
		v.push_back(Point2d(272, 148));
		v.push_back(Point2d(796, 157));
		v.push_back(Point2d(828, 522));
		v.push_back(Point2d(226, 514));
	}

	pontosDoCanto.push_back(v);
	aux = originalImages[4].clone();
	calculaGridROI(aux, pontosDoCanto[4], 4);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[4]);
		waitKey(0);
		cv::destroyWindow(imagePaths[4]);
	}
}

void selecionarCantosComMouse(){
	// Selecionar 4 pontos limite do padrao para cada foto
	Mat aux;
	for (int i = 0; i < imagePaths.size(); i++){
		countPoints = 0;
		mostraImagem(originalImages[i], imagePaths[i]);
		setMouseCallback(imagePaths[i], onMouseCallBack, &i);

		while (countPoints < 4){
			waitKey(20);
		}

		setMouseCallback(imagePaths[i], NULL, NULL);

		//calcula o grid	
		aux = originalImages[i].clone();
		calculaGridROI(aux, pontosDoCanto[i], i);
		mostraImagem(aux, imagePaths[i]);
		waitKey(0);

		cv::destroyWindow(imagePaths[i]);
	}
}

void onMouseCallBack(int e, int x, int y, int flags, void * imageId){
	if (e == CV_EVENT_LBUTTONDOWN){
		int id = *((int*)imageId);
		//printf("x,y (%d,%d)\n", x, y);
		printf("v.push_back(Point2d(%d,%d));\n", x, y);

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
			desenhaCruz(aux, x, y, Scalar(0, 0, 255));
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

void imprimeMat3xn(Mat m, int s){
	cout << "[" << endl;
	for (int j = 0; j < s; j++){
		cout << " " << m.at<double>(0, j) << ", " << m.at<double>(1, j) << ", " << m.at<double>(2, j) << endl;
	}
	cout << "]" << endl;
}

void calculaGridROI(Mat img, vector<Point2d> pontosCanto, int id){
	vector<Point2f> planePoints, imagePoints;

	planePoints.push_back(Point2f(0, 0));
	planePoints.push_back(Point2f(1, 0));
	planePoints.push_back(Point2f(1, 1));
	planePoints.push_back(Point2f(0, 1));

	imagePoints.push_back(Point2f(pontosCanto[0].x, pontosCanto[0].y));
	imagePoints.push_back(Point2f(pontosCanto[1].x, pontosCanto[1].y));
	imagePoints.push_back(Point2f(pontosCanto[2].x, pontosCanto[2].y));
	imagePoints.push_back(Point2f(pontosCanto[3].x, pontosCanto[3].y));

	Mat H = findHomography(planePoints, imagePoints);

	if (MOSTRA_GRID)
		cout << "H = " << endl << " " << H << endl << endl;

	int quantPontosGrid = (nHorizontal + 1) * (nVertical + 1);
	Mat pontosGridPlano(3, quantPontosGrid, CV_64F);
	Mat pontosGridImagem(3, quantPontosGrid, CV_64F);
	vector<Point2f> pontos2dImagem(quantPontosGrid);
	vector<Point2f> aux;

	float x, y;

	//calculando pontos no plano
	for (int i = 0; i < nHorizontal + 1; i++){
		x = (float)i / (float)nHorizontal;
		for (int j = 0; j < nVertical + 1; j++){
			y = (float)j / (float)nVertical;
			aux.push_back(Point2f(x, y));
		}
	}
	//cout << "aux = " << endl << " " << aux << endl;
	//passa do vetor para a matriz

	for (int j = 0; j < quantPontosGrid; j++){
		pontosGridPlano.at<double>(0, j) = aux[j].x;
		pontosGridPlano.at<double>(1, j) = aux[j].y;
		pontosGridPlano.at<double>(2, j) = 1;
	}

	//multiplica pela homographia	
	pontosGridImagem = H * pontosGridPlano;

	//passa para o vetor
	for (int j = 0; j < quantPontosGrid; j++){
		//dividir coordenadas homogeneas pelo z
		pontos2dImagem[j] = Point2f(pontosGridImagem.at<double>(0, j) / pontosGridImagem.at<double>(2, j),
			pontosGridImagem.at<double>(1, j) / pontosGridImagem.at<double>(2, j));
		// coloca cruzes nos pontos mapeados
		//desenhaCruz(img, pontos2dImagem[j].x, pontos2dImagem[j].y, Scalar(255, 255, 0));
	}

	// Imprime grid
	if (MOSTRA_GRID){
		//linhas horizontais
		x = (nVertical + 1)*(nHorizontal);
		for (int i = 0; i < nVertical + 1; i++){
			desenhaLinha(img, pontos2dImagem[i],
				pontos2dImagem[i + x], Scalar(255, 0, 0));
		}
		//linhas verticais
		for (int i = 0; i < nHorizontal + 1; i++){
			desenhaLinha(img, pontos2dImagem[i*(nVertical + 1)],
				pontos2dImagem[i*(nVertical + 1) + nVertical], Scalar(255, 0, 0));
		}
	}

	pontosGrid.push_back(pontos2dImagem);
}

void mostraImagem(Mat img, string name){
	namedWindow(name);
	imshow(name, img);
	moveWindow(name, 150, 150);
}

int _ind(int i, int j){
	return j*(nVertical + 1) + i;
}

Point2d getMenorPonto(Point2f p[4]){
	Point2f c(999999999, 999999999);
	for (int i = 0; i < 4; i++){
		if (p[i].x < c.x) c.x = p[i].x;
		if (p[i].y < c.y) c.y = p[i].y;
	}
	return c;
}

Point2d getMaiorPonto(Point2f p[4]){
	Point2f c(0, 0);
	for (int i = 0; i < 4; i++){
		if (p[i].x > c.x) c.x = p[i].x;
		if (p[i].y > c.y) c.y = p[i].y;
	}
	return c;
}

void desenhaRetangulo(Mat img, Point p1, Point p2, Scalar c){
	rectangle(img, p1, p2, c);
}

void desenhaQuadrilatero(Mat img, Point p[4], Scalar c){
	line(img, p[0], p[1], c);
	line(img, p[1], p[2], c);
	line(img, p[2], p[3], c);
	line(img, p[3], p[0], c);
}

void computeEllipse(Mat imgOriginal, int idImg){
	//Arredonda os pontos do grid de float para inteiro
	vector<Point2f> gridFloat = pontosGrid[idImg];
	/*vector<Point2d> gridInt(gridFloat.size());
	for (int i = 0; i < gridFloat.size(); i++){
		gridInt[i].x = round(gridFloat[i].x);
		gridInt[i].y = round(gridFloat[i].y);
	}*/

	Mat imgDoLoop;
	double min, max, cweight, thresh;
	int countPixelsInCircle, sumX, sumY;
	Point2f centro;
	vector<Point2f> posicaoPontosDeControle;

	//extrair cada regiao do grid e calcular centro da elipse
	for (int i = 0; i < nVertical; i++){
		for (int j = 0; j < nHorizontal; j++){
			Point2f p[4];
			p[0] = gridFloat[_ind(i, j)];
			p[1] = gridFloat[_ind(i, j + 1)];
			p[2] = gridFloat[_ind(i + 1, j + 1)];
			p[3] = gridFloat[_ind(i + 1, j)];

			imgDoLoop = imgOriginal.clone();

			Point2f c1 = getMenorPonto(p);
			desenhaCruz(imgDoLoop, c1.x, c1.y, Scalar(0, 0, 255));
			Point2f c2 = getMaiorPonto(p);
			desenhaCruz(imgDoLoop, c2.x, c2.y, Scalar(255, 0, 0));
			int w = c2.x - c1.x, h = c2.y - c1.y;

			Mat cellImg(imgOriginal, Rect(c1.x, c1.y, w, h));

			//threshold it
			Mat cellImgBW(cellImg.rows, cellImg.rows, CV_8UC1);
			cvtColor(cellImg, cellImgBW, CV_RGB2GRAY);
			minMaxLoc(cellImgBW, &min, &max);
			cweight = 0.5;
			thresh = min*cweight + max*(1 - cweight);
			//calcular a media da posicao dos pontos no circulo e fazer o threshold
			countPixelsInCircle = sumX = sumY = 0;
			for (int k = 0; k < cellImgBW.cols; k++){
				for (int l = 0; l < cellImgBW.rows; l++){
					if (cellImgBW.at<uchar>(l, k)>thresh){// na imagem o circulo eh preto
						cellImgBW.at<uchar>(l, k) = 255;
					}
					else{
						cellImgBW.at<uchar>(l, k) = 0;
						countPixelsInCircle++;
						sumX += k;
						sumY += l;
					}
				}
			}
			centro.x = ((float)sumX / (float)countPixelsInCircle);
			centro.y = ((float)sumY / (float)countPixelsInCircle);

			desenhaCruz(imgDoLoop, centro.x + c1.x, centro.y + c1.y, Scalar(255, 255, 255));
			desenhaRetangulo(imgDoLoop, c1, c2, Scalar(255, 255, 0));
			//desenhaQuadrilatero(imgDoLoop, p, Scalar(0, 0, 255));

			if (MOSTRA_CADA_ROI){
				mostraImagem(imgDoLoop, "ROI");
				mostraImagem(cellImgBW, "Celula");
				moveWindow("Celula", 1000, 150);
				waitKey(0);
			}

			posicaoPontosDeControle.push_back(centro + c1);

		}//for
	}//for

	centrosOriginal.push_back(posicaoPontosDeControle);

	cv::destroyWindow("Celula");
	cv::destroyWindow("ROI");

}

void calcPosicoesIdeaisObjeto(){
	vector<Point3f> v;
	centrosIdealObjeto.clear();
	for (int i = 0; i < nVertical; ++i)
	for (int j = 0; j < nHorizontal; ++j)
		v.push_back(Point3f(float(i*distanceCP), float(j*distanceCP), 0));

	centrosIdealObjeto.push_back(v);
}

static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	Mat img;
	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);

		if (MOSTRA_REPROJECAO){
			img = originalImages[i].clone();

			for (int k = 0; k < nVertical; k++){
				for (int j = 0; j < nHorizontal; j++){
					desenhaCruz(img, imagePoints[i][k*nHorizontal + j].x,
						imagePoints[i][k*nHorizontal + j].y, Scalar(255, 255, 255));
					desenhaCruz(img, imagePoints2[k*nHorizontal + j].x,
						imagePoints2[k*nHorizontal + j].y, Scalar(0, 0, 255));
				}
			}
			mostraImagem(img, "Reprojecao");
			moveWindow("Reprojecao", 150, 150);
			waitKey(0);

		}
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}
	cv::destroyWindow("Reprojecao");
	return std::sqrt(totalErr / totalPoints);
}

void print(Mat mat, int prec)
{
	for (int i = 0; i < mat.size().height; i++)
	{
		cout << "[";
		for (int j = 0; j < mat.size().width; j++)
		{
			cout << fixed << setprecision(prec) << mat.at<double>(i, j);
			if (j != mat.size().width - 1)
				cout << ", ";
			else
				cout << "]" << endl;
		}
	}
	cout << endl;
}

void calibraCamera(){
	//Find intrinsic and extrinsic camera parameters
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flag) cameraMatrix.at<double>(0, 0) = 1.0;

	distCoeffs = Mat::zeros(8, 1, CV_64F);
	Size imageSize(originalImages[0].size());

	calcPosicoesIdeaisObjeto();
	//cout << "posicao objeto: " << centrosIdealObjeto[0] << endl;

	centrosIdealObjeto.resize(originalImages.size(), centrosIdealObjeto[0]);

	double rms = calibrateCamera(centrosIdealObjeto, centrosOriginal,
		imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flag);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	vector<float> reprojErrs;
	double totalAvgErr = 0;
	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(centrosIdealObjeto, centrosOriginal,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	if (ok){
		cout << "Calibration succeeded. avg re projection error = " << totalAvgErr << endl;
		cout << endl << "Camera Matrix (intrinsic) " << endl;
		print(cameraMatrix, 2);
		cout << endl << "Coeficientes de Distorcao " << endl;
		print(distCoeffs, 2);

		/*for (int i = 0; i < rvecs.size(); i++){
			cout << endl << "R-" << i << endl;
			print(rvecs[i], 2);
			cout << endl << "t-" << i << endl;
			print(tvecs[i], 2);
			cout << endl;
		}*/
	}
	else{
		cout << "Calibration failed. avg re projection error = " << totalAvgErr;
	}
}

void undistortImages(){
	undistortedImages.clear();
	centrosUndistorted.clear();

	// Undistort imagens
	Mat imageUndistorted, aux, aux2;
	for (int i = 0; i < imagePaths.size(); i++){
		aux = originalImages[i];

		undistort(aux, imageUndistorted, cameraMatrix, distCoeffs);
		undistortedImages.push_back(imageUndistorted);

		//distorce os centros dos pontos de controle tbm
		vector<Point2f> centrosUndist;
		undistortPoints(centrosOriginal[i], centrosUndist, cameraMatrix, distCoeffs, noArray(), cameraMatrix);
		centrosUndistorted.push_back(centrosUndist);

		if (MOSTRA_UNDISTORTED){
			//coloca cruz para a imagem original 
			desenhaCruz(aux, centrosOriginal[i][0].x, centrosOriginal[i][0].y, Scalar(255, 255, 255));
			desenhaCruz(aux, centrosOriginal[i][9].x, centrosOriginal[i][9].y, Scalar(255, 255, 255));
			desenhaCruz(aux, centrosOriginal[i][69].x, centrosOriginal[i][69].y, Scalar(255, 255, 255));
			desenhaCruz(aux, centrosOriginal[i][60].x, centrosOriginal[i][60].y, Scalar(255, 255, 255));

			resize(aux, aux, Size(512, 384));
			mostraImagem(aux, "Image original");
			moveWindow("Image original", 0, 0);

			//coloca criz nos centros sem a distorcao
			aux2 = imageUndistorted.clone();
			for (int m = 0; m < nVertical; m++){
				for (int n = 0; n < nHorizontal; n++){
					desenhaCruz(aux2, centrosUndist[m*nHorizontal + n].x,
						centrosUndist[m*nHorizontal + n].y, Scalar(255, 255, 255));
				}
			}
			resize(aux2, aux2, Size(512, 384));
			mostraImagem(aux2, "Image undistorted");
			moveWindow("Image undistorted", 530, 0);
			waitKey(0);
		}

		//nao sei pq nao funciona se nao redimensionar a imageUndistorted
		resize(imageUndistorted, imageUndistorted, Size(512, 384));

	}
	cv::destroyWindow("Image original");
	cv::destroyWindow("Image undistorted");
}

void frontoParalelo(){
	// Input Quadilateral or Image plane coordinates
	Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];

	// Lambda Matrix
	Mat perspectiveTransformMatrix;// (2, 4, CV_32FC1);
	//Input and Output Image;
	Mat outputImage, inputImage;

	float div = 10;
	float d1w, d1h; //distancia entre cantos horizontal e vertical
	float offsetx = 60, offsety = 90;
	distanciaCentro = -1;

	frontoParallelImages.clear();
	homografias.clear();
	centrosFrontoParallel.clear();

	for (int i = 0; i < imagePaths.size(); i++)
	{
		inputImage = undistortedImages[i].clone();
		//perspectiveTransformMatrix = Mat::zeros(inputImage.rows, inputImage.cols, inputImage.type());

		// The 4 points that select quadilateral on the inputImage , from top-left in clockwise order
		// These four pts are the sides of the rect box used as inputImage 
		inputQuad[0] = centrosUndistorted[i][0];
		inputQuad[1] = centrosUndistorted[i][nHorizontal-1];
		inputQuad[2] = centrosUndistorted[i][nVertical*nHorizontal-1];
		inputQuad[3] = centrosUndistorted[i][(nVertical-1)*nHorizontal];

		// The 4 points where the mapping is to be done , from top-left in clockwise order

		/*outputQuad[0] = Point2f(inputImage.cols / div, inputImage.rows / div);
		outputQuad[1] = Point2f((inputImage.cols * (div - 1)) / div, inputImage.rows / div);
		outputQuad[2] = Point2f((inputImage.cols * (div - 1)) / div, (inputImage.rows * (div - 1)) / div);
		outputQuad[3] = Point2f(inputImage.cols / div, (inputImage.rows * (div - 1)) / div);*/

		//achar os 4 pontos de forma correta (ainda nao foi)
		outputQuad[0] = Point2f(offsetx, offsety);
		outputQuad[1] = Point2f(inputImage.cols - offsetx, offsety);
		outputQuad[2] = Point2f(inputImage.cols - offsetx, inputImage.rows - offsety);
		outputQuad[3] = Point2f(offsetx, inputImage.rows - offsety);


		// Get the Perspective Transform Matrix i.e. lambda 
		perspectiveTransformMatrix = getPerspectiveTransform(inputQuad, outputQuad);

		// Apply the Perspective Transform just found to the src image
		warpPerspective(inputImage, outputImage, perspectiveTransformMatrix, Size(inputImage.cols, inputImage.rows)/*outputImage.size()*/);

		frontoParallelImages.push_back(outputImage);

		//salvar homografia
		homografias.push_back(perspectiveTransformMatrix);

		//calcular os centros no plano fronto paralelo tbm
		vector<Point2f> centrosNoFrontoParalelo;
		perspectiveTransform(centrosUndistorted[i], centrosNoFrontoParalelo, perspectiveTransformMatrix);
		centrosFrontoParallel.push_back(centrosNoFrontoParalelo);

		if (i == 1){
			distanciaCentro = norm(centrosUndistorted[i][1] - centrosUndistorted[i][0]);
		}

		if (MOSTRA_PERSPECTIVE_TRANSFORM)
		{
			//desenhaCentros(inputImage, centrosUndistorted[i]);
			resize(inputImage, inputImage, Size(512, 384));
			mostraImagem(inputImage, "Image undistorted 2");
			moveWindow("Image undistorted 2", 0, 0);

			//desenhaCentros(outputImage, centrosFrontoParallel[i]);
			resize(outputImage, outputImage, Size(512, 384));
			mostraImagem(outputImage, "Image fronto parallel");
			moveWindow("Image fronto parallel", 530, 0);

			cout << "Perspective matrix " << endl;
			print(perspectiveTransformMatrix, 3);

			waitKey(0);

		}// end mostra

	}//end  for (int i = 0; i < imagePaths.size(); i++)
	cv::destroyWindow("Image undistorted 2");
	cv::destroyWindow("Image fronto parallel");

}// end void frontoParalelo()

void findCentrosFrontoParalelo(){
	Mat aux;
	vector<Point2f> novosCentros;
	
	novosCentrosFrontoParallel.clear();
	for (int i = 0; i < frontoParallelImages.size(); i++){
		aux = frontoParallelImages[i].clone();

		//acha centros na imagem frontal
		if (AS_ANKUR)
			novosCentros = computeCorrelationSSD(aux, centrosFrontoParallel[i]);
		else
			novosCentros = fitEllipse(aux, centrosFrontoParallel[i]);

		novosCentrosFrontoParallel.push_back(novosCentros);
	}
}

//como no ankur (matlab)
vector<Point2f> computeCorrelationSSD(Mat image, vector<Point2f> centros){
	vector<Point2f> novosCentros;
	Mat circulo = imread("imagens/gcircleFilter.bmp");

	int dist = 97;// (int)distanciaCentro;

	resize(circulo, circulo, Size(dist, dist));
	cvtColor(circulo, circulo, CV_BGR2GRAY);
	//circulo = 255 - circulo;


	/*Mat ssd(100, 100, CV_64F, Scalar(0));
	Point2d p1(50, 50), p2(50, 49), l, f;
	ssd.at<double>(p1.x, p1.y) = 255;

	while (true){
	ssd.at<double>(p2.x, p2.y) = 255;

	l.x = p1.y - p2.y;
	l.y = p2.x - p1.x;
	f = p2 - p1;

	p1 = p2;
	if (ssd.at<double>(l.x + p2.x, l.y + p2.y) == 0)
	p2 = l + p2;
	else
	p2 = f + p2;

	mostraImagem(ssd, "ssd");
	cv::waitKey(1);
	}*/

	Mat aux;
	// para cada ponto de controle (centro)
	cout << "Procurando pontos de controle ..." << endl;
	for (int v = 0; v < nVertical; v++){
		for (int h = 0; h < nHorizontal; h++){

			aux = image.clone();
			cvtColor(aux, aux, CV_BGR2GRAY);

			//verifica limites da imagem
			int roi_x = centros[v*nHorizontal + h].x - (dist / 2 + 5);
			int roi_w = (dist / 2 + 5) * 2;
			if (roi_x < 0){
				roi_w += roi_x;
				roi_x = 0;
			}
			if (roi_x + roi_w >= image.cols - 1){
				roi_w = image.cols - roi_x;
			}
			int roi_y = centros[v*nHorizontal + h].y - (dist / 2 + 5);
			int roi_h = (dist / 2 + 5) * 2;
			if (roi_y < 0){
				roi_h += roi_y;
				roi_y = 0;
			}
			if (roi_y + roi_h >= image.rows - 1){
				roi_h = image.rows - roi_y;
			}

			/*desenhaRetangulo(aux, Point2f(roi_x, roi_y), Point2f(roi_x + roi_w, roi_y + roi_h), Scalar(255, 255, 255));
			mostraImagem(aux, "roi");
			waitKey(0);
			continue;*/

			/*Mat ssd(roi_h, roi_w, CV_64F, 999999999999);
			double* p = ssd.ptr<double>(10);
			for (int i = 0; i < 20; i++){
			p[i] = 50;
			}
			mostraImagem(ssd, "ssd");
			cv::waitKey(0);*/


			float menor = 999999999999;
			Point2f menorPosition;
			int a = 1;


			//itera na roi
			for (int j = roi_y; j < roi_h + roi_y; j++){
				for (int i = roi_x; i < roi_w + roi_x; i++){

					float sum = 0;
					if (i + dist > roi_w + roi_x || j + dist > roi_h + roi_y) break;

					//itera na subroi
					uchar * p1, *p2;
					for (int y = j; y < j + dist; y++){
						p1 = aux.ptr<uchar>(y);
						p2 = circulo.ptr<uchar>(y - j);
						for (int x = i; x < i + dist; x++){
							//sum += pow(aux.at<uchar>(y, x) - circulo.at<uchar>(y - j, x - i), 2);
							sum += pow(p1[x] - p2[x - i], 2);
						}
					}

					if (a == 1) {
						menor = sum;
						a = 0;
					}
					if (sum < menor){
						menor = sum;
						menorPosition.x = i;
						menorPosition.y = j;
					}
					//ssd.at<double>(j - roi_y, i - roi_x) = sum;
				}
			}// end - for itera roi



			//itera na roi usando iteracao espiral		
			//Mat ssd(30, 30, CV_64F, Scalar(0));
			//Point2d p1(15, 15), p2(15, 14), l, f, c(15, 15), 
			//	cur_cen(centros[v*nHorizontal + h].x - c.x, centros[v*nHorizontal + h].y - c.y);
			//ssd.at<double>(p1.x, p1.y) = 255;

			//int max = 200;

			//while (max--){
			//	ssd.at<double>(p2.x, p2.y) = 255;

			//	//itera na subroi	
			//	float sum = 0;
			//	for (int y = 0; y < dist; y++){
			//		for (int x = 0; x < dist; x++){
			//			sum += pow(aux.at<uchar>(p2.x + cur_cen.x - dist / 2, p2.y + cur_cen.y - dist / 2)
			//				- circulo.at<uchar>(y, x), 2);
			//		}
			//	}
			//	if (a == 1) {
			//		menor = sum;
			//		a = 0;
			//	}
			//	if (sum < menor){
			//		menor = sum;
			//		menorPosition.x = p2.x + cur_cen.x;
			//		menorPosition.y = p2.y + cur_cen.y;
			//	}
			//	
			//	//incremento
			//	l.x = p1.y - p2.y;
			//	l.y = p2.x - p1.x;
			//	f = p2 - p1;

			//	p1 = p2;
			//	if (ssd.at<double>(l.x + p2.x, l.y + p2.y) == 0)
			//		p2 = l + p2;
			//	else
			//		p2 = f + p2;

			//}

			//normal
			//cout << "old " << Point2d(centros[v*nHorizontal + h].x, centros[v*nHorizontal + h].y) << endl;
			//cout << "new " << Point2d(menorPosition.x + dist / 2, menorPosition.y + dist / 2) << endl;

			/*desenhaCruz(image, centros[v*nHorizontal + h].x, centros[v*nHorizontal + h].y, Scalar(0, 0, 255));
			desenhaCruz(image, menorPosition.x + dist / 2, menorPosition.y + dist / 2, Scalar(255, 255, 255));
			mostraImagem(image, "Novos centros");
			cv::waitKey(0);*/

			novosCentros.push_back(Point2f(menorPosition.x + dist / 2, menorPosition.y + dist / 2));
			//cout << " ... achou " << v << "," << h << " ... " << endl;


			//espiral
			/*cout << "old " << Point2d(centros[v*nHorizontal + h].x, centros[v*nHorizontal + h].y) << endl;
			cout << "new " << Point2d(menorPosition.x, menorPosition.y) << endl;

			desenhaCruz(image, centros[v*nHorizontal + h].x, centros[v*nHorizontal + h].y, Scalar(0, 0, 255));
			desenhaCruz(image, menorPosition.x, menorPosition.y, Scalar(255, 255, 255));
			mostraImagem(image, "Novos centros");
			cv::waitKey(0);

			novosCentros.push_back(Point2f(menorPosition.x, menorPosition.y));
			cout << " ... achou " << v << "," << h << " ... " << endl;*/

		} //for - centros
	}//for - centros
	cout << "     ... end "<< endl;
	if (SHOW_NOVOS_CENTROS_SSD){
		desenhaCentros(image, novosCentros);
		mostraImagem(image, "Novos centros");
		cv::waitKey(0);
		cv::destroyWindow("Novos centros");
	}	

	return novosCentros;
}

//como no praksh
vector<Point2f> fitEllipse(Mat image, vector<Point2f> centros){
	vector<Point2f> novosCentros;
	Mat aux;
	int dist = (int)distanciaCentro;
	Mat roi;
	cout << "Procurando pontos de controle ..." << endl;
	for (int v = 0; v < nVertical; v++){
		for (int h = 0; h < nHorizontal; h++){
			aux = image.clone();
			cvtColor(aux, aux, CV_BGR2GRAY);

			//verifica limites da imagem
			int roi_x = centros[v*nHorizontal + h].x - (dist / 2 + 20);
			int roi_w = (dist / 2 + 20) * 2;
			if (roi_x < 0){
				roi_w += roi_x;
				roi_x = 0;
			}
			if (roi_x + roi_w >= image.cols - 1){
				roi_w = image.cols - roi_x;
			}
			int roi_y = centros[v*nHorizontal + h].y - (dist / 2 + 20);
			int roi_h = (dist / 2 + 20) * 2;
			if (roi_y < 0){
				roi_h += roi_y;
				roi_y = 0;
			}
			if (roi_y + roi_h >= image.rows - 1){
				roi_h = image.rows - roi_y;
			}
			
			roi = aux(Rect(roi_x, roi_y, roi_w, roi_h));

			/*Mat hist;
			float range[] = { 0, 256 };
			int histSize = 256;
			const float* histRange = { range };
			calcHist(&roi, 1, 0, Mat(), hist, 1, &histSize, &histRange);
			showHistogram(hist, histSize);*/

			// threshold adaptativo
			float media=0;
			int cont = 0;

			//itera na roi acha media inicial			
			for (int j = roi_y; j < roi_h + roi_y; j++){
				for (int i = roi_x; i < roi_w + roi_x; i++){
					media += roi.at<uchar>(j - roi_y, i - roi_x);					
				}
			}
			media /= (float)roi_w * roi_h;
			//cout << "media " << media << endl;

			float m = 0;
			cont = 0;
			//itera na roi acha media inicial			
			for (int j = roi_y; j < roi_h + roi_y; j++){
				for (int i = roi_x; i < roi_w + roi_x; i++){
					if (roi.at<uchar>(j - roi_y, i - roi_x)<=media){
						m += roi.at<uchar>(j - roi_y, i - roi_x);
						cont++;
					}						
				}
			}
			m /= (float)cont;
			//cout << "m " << m << endl;

			float t = m, m1, m2;
			int passo = 10, cont1, cont2;
			while (passo--){
				//itera na roi acha media inicial			
				//calcula m1 e m2
				cont1 = 0;
				cont2 = 0;
				m1 = 0;
				m2 = 0;
				for (int j = roi_y; j < roi_h + roi_y; j++){
					for (int i = roi_x; i < roi_w + roi_x; i++){
						if (roi.at<uchar>(j - roi_y, i - roi_x) <= media){
							if (roi.at<uchar>(j - roi_y, i - roi_x) <= t){
								m1 += roi.at<uchar>(j - roi_y, i - roi_x);
								cont1++;
							}
							else{
								m2 += roi.at<uchar>(j - roi_y, i - roi_x);
								cont2++;
							}
						}
					}
				}
				//cout << "diff " << abs(t - (m1 / cont1 + m2 / cont2) / 2) << endl;
				t = (m1/cont1 + m2/cont2) / 2;
				//cout << "t " << t << endl;
			}

			// aplica o threshold em roi
			for (int j = roi_y; j < roi_h + roi_y; j++){
				for (int i = roi_x; i < roi_w + roi_x; i++){
					if (roi.at<uchar>(j - roi_y, i - roi_x) <= t){
						roi.at<uchar>(j - roi_y, i - roi_x) = 0;
					}else
						roi.at<uchar>(j - roi_y, i - roi_x) = 255;
				}
			}

			//detecta borda 
			Mat threshold_output;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			/// Detect edges using Threshold
			threshold(roi, threshold_output, t, 255, THRESH_BINARY);
			findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			//fit ellipse
			vector<RotatedRect> minEllipse(contours.size());
			for (int i = 0; i < contours.size(); i++)
			{
				if (contours[i].size() > 5)
				{
					minEllipse[i] = fitEllipse(Mat(contours[i]));
				}
			}

			/// Draw contours + rotated rects + ellipses
			Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
			for (int i = 0; i< contours.size(); i++)
			{
				Scalar color = Scalar(255, 255, 255);
				// contour
				drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
				// ellipse
				ellipse(drawing, minEllipse[i], color, 2, 8);	
				//centro 
				desenhaCruz(drawing, minEllipse[1].center.x, minEllipse[1].center.y, Scalar(0, 0, 255));
			}
			
			/// Show in a window
			/*namedWindow("Contours", CV_WINDOW_AUTOSIZE);
			imshow("Contours", drawing);

			cout << "centro x,y: "<< minEllipse[1].center.x <<", "<< minEllipse[1].center.y << endl;

			desenhaRetangulo(aux, Point2f(roi_x, roi_y), Point2f(roi_x + roi_w, roi_y + roi_h),Scalar(255,255,255));
			mostraImagem(aux,"fit");
			mostraImagem(roi, "roi");
			waitKey(0);*/
			
			novosCentros.push_back(minEllipse[1].center + Point2f(roi_x, roi_y));
		}
	}
	cout << "... achou ..." << endl;
	if (SHOW_NOVOS_CENTROS_ELLIPSE){
		desenhaCentros(image, novosCentros, Scalar(255, 0, 255));
		mostraImagem(image, "Novos centros");
		cv::waitKey(0);
		cv::destroyWindow("Novos centros");
	}	

	return novosCentros;
}

void desenhaCentros(Mat img, vector<Point2f> centros){
	for (int v = 0; v < nVertical; v++){
		for (int h = 0; h < nHorizontal; h++){
			desenhaCruz(img, centros[v*nHorizontal + h].x,
				centros[v*nHorizontal + h].y, Scalar(0, 255, 255));
		}
	}
}

void desenhaCentros(Mat img, vector<Point2f> centros, Scalar cor){
	for (int v = 0; v < nVertical; v++){
		for (int h = 0; h < nHorizontal; h++){
			desenhaCruz(img, centros[v*nHorizontal + h].x,
				centros[v*nHorizontal + h].y, cor);
		}
	}
}

Point2f distorcePonto(Point2f point)
{
	double cx = cameraMatrix.at<double>(0, 2);
	double cy = cameraMatrix.at<double>(1, 2);
	double fx = cameraMatrix.at<double>(0, 0);
	double fy = cameraMatrix.at<double>(1, 1);

	// To relative coordinates <- this is the step you are missing.
	double x = (point.x - cx) / fx;
	double y = (point.y - cy) / fy;

	double r2 = x*x + y*y;

	double k1 = distCoeffs.at<double>(0, 0);
	double k2 = distCoeffs.at<double>(1, 0);
	double k3 = distCoeffs.at<double>(4, 0);
	double p1 = distCoeffs.at<double>(2, 0);
	double p2 = distCoeffs.at<double>(3, 0);

	// Radial distorsion
	double xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
	double yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

	// Tangential distorsion
	xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
	yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

	// Back to absolute coordinates.
	xDistort = xDistort * fx + cx;
	yDistort = yDistort * fy + cy;

	return Point2f((float)xDistort, (float)yDistort);
}

void proj_dist_centros(){
	Mat aux;

	centrosProjetados.clear();
	//projeta
	for (int i = 0; i < undistortedImages.size(); i++){
		aux = undistortedImages[i].clone();

		vector<Point2f> centrosProj;
		perspectiveTransform(novosCentrosFrontoParallel[i], centrosProj, homografias[i].inv());

		desenhaCentros(aux, centrosUndistorted[i], Scalar(0, 255, 255));
		desenhaCentros(aux, centrosProj, Scalar(255, 255, 255));

		centrosProjetados.push_back(centrosProj);

		mostraImagem(aux, "Proj Centros");
		waitKey(0);
	}
	cv::destroyWindow("Proj Centros");

	centrosDistorcidos.clear();
	//distorce
	for (int i = 0; i < originalImages.size(); i++){
		aux = originalImages[i].clone();

		vector<Point2f> centrosDist;

		cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
		rVec.at<double>(0) = 0;
		rVec.at<double>(1) = 0;
		rVec.at<double>(2) = 0;
		cv::Mat tVec(3, 1, cv::DataType<double>::type); // Translation vector
		tVec.at<double>(0) = 0;
		tVec.at<double>(1) = 0;
		tVec.at<double>(2) = 0;

		/*vector<Point3f> v;
		for (int i = 0; i < centrosProj.size(); ++i)
			v.push_back(Point3f(centrosProj[i].x, centrosProj[i].y, 0));

		projectPoints(v, rVec, tVec, cameraMatrix, distCoeffs, centrosDist);*/

		for (int j = 0; j < centrosProjetados[i].size(); j++)			
			centrosDist.push_back(distorcePonto(centrosProjetados[i][j]));
		
		centrosDistorcidos.push_back(centrosDist);
		
		desenhaCentros(aux, centrosDist, Scalar(255, 255, 255));
		desenhaCentros(aux, centrosOriginal[i], Scalar(0, 255, 255));

		//atualiza o centro original para os novos centros, pq a calibracao vai rodar com eles
		centrosOriginal[i] = centrosDist;

		mostraImagem(aux, "Dist centros");
		waitKey(0);
	}
	cv::destroyWindow("Dist centros");
}

void showHistogram(Mat b_hist, int histSize){
	cout << b_hist << endl;

	// Draw the histograms for roi
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	
}

// end - Sasha Nicolas


//tentativa do fronto paralelo - NOT WORKING
//if (MOSTRA_UNPROJECTED)
//{
//	Mat view, rview, map1, map2;
//	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
//		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, originalImages[0].size(), 1, originalImages[0].size(), 0),
//		originalImages[0].size(), CV_16SC2, map1, map2);
//
//	for (int i = 0; i < imagePaths.size(); i++)
//	{
//		view = originalImages[i].clone();
//		//resize(view, view, Size(1000, 750));
//		if (view.empty())
//			continue;
//		remap(view, rview, map1, map2, INTER_LINEAR);
//
//		mostraImagem(rview, "Image View");
//
//		waitKey(0);
//	}
//	cv::destroyWindow("Image View");
//}
