// Copyright (c) 2010-2011, Tuji
// All rights reserved.
// 
// ${license}
//
// Author: LIU Yi

#include "evaluate/icdar2005.h"

#include <opencv2/highgui/highgui.hpp>

#include "evaluate/evaluate.h"
#include "common/core.h"
#include "xml/tinyxml.h"
#include "common/test.h"

using namespace std;
using namespace cv;
using namespace dtxt;

ICDAR2005::~ICDAR2005() {
  map<string, list<TextLine*>*>::iterator mit = T_list_map_.begin();
  for (; mit != T_list_map_.end(); ++mit) {
    list<TextLine*>::iterator lit = mit->second->begin(), lend = mit->second
        ->end();
    for (; lit != lend; ++lit) {
      delete *lit;
    }
    delete mit->second;
  }
}

list<TextLine*>* ICDAR2005::RetrieveTgtList(const string& img_path) {
  size_t pos = img_path.find_last_of('/', img_path.find_last_of('/'));
  string id = img_path.substr(pos + 1);

  return T_list_map_.at(id);
}

void ICDAR2005::InitTListMap() {
  string xml_path = test_data_dir() + "/locations.xml";
  TiXmlDocument doc(xml_path.c_str());
  doc.LoadFile();

  TiXmlElement* tagset_elem = doc.FirstChildElement();
  TiXmlElement* image_elem = tagset_elem->FirstChildElement();
  for (; image_elem != NULL; image_elem = image_elem->NextSiblingElement()) {
    string image_name = image_elem->FirstChildElement("imageName")->GetText();

    TiXmlElement* rects_elem = image_elem->FirstChildElement(
        "taggedRectangles");

    list<TextLine*>* rect_list = new list<TextLine*>;
    TiXmlElement* rect_elem = rects_elem->FirstChildElement();
    for (; rect_elem != NULL; rect_elem = rect_elem->NextSiblingElement()) {
      int x = static_cast<int>(atof(rect_elem->Attribute("x")));
      int y = static_cast<int>(atof(rect_elem->Attribute("y")));
      int width = static_cast<int>(atof(rect_elem->Attribute("width")));
      int height = static_cast<int>(atof(rect_elem->Attribute("height")));

      rect_list->push_back(new TextLine(x, y, x + width - 1, y + height - 1));
    }

    T_list_map_.insert(pair<string, list<TextLine*>*>(image_name, rect_list));
  }
}

void ICDAR2005::RetrieveImgIds(std::vector<std::string>* idlist) {
  throw "ICDAR2005::RetrieveImgIds is unimplemented yet";
}
